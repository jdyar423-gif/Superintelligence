"""PointerXL: streaming-memory Transformer LM with a learned long-range pointer (copy) head.

Design (targets WT2's dominant difficulty: rare entities that repeat 1-32 KB apart):
  * Streaming recurrent memory (Transformer-XL style): the model reads contiguous streams segment by
    segment and every layer attends to a detached K/V memory of the previous `mem_len` tokens.
    Keys are stored un-rotated; RoPE is applied at use time with consistent relative positions.
  * Learned pointer-sentinel head over a long memory (`ptr_len` tokens) of final hidden states:
        p(y_t) = a_sent(t) * softmax(z_t)[y_t] + sum_{i < t, next(i) = y_t} a_i(t)
    where (a_1..a_n, a_sent) = softmax([q_t . k_i ..., s_t]).  Values are the *next tokens* of
    memory positions (Grave-cache style pairs), so the head directly copies continuations of earlier
    contexts. It is trained end-to-end with the mixture NLL; at test time it is a proper normalised
    distribution and strictly causal (candidates are only already-seen positions).
  * Modern block: RMSNorm, QK-norm, partial RoPE + key offset (1-layer induction), per-head sigmoid
    output gate, value residual, U-Net skips, x0 re-injection, SmearGate, ReLU^2 MLP, logit softcap,
    tied + normalised embeddings.
All memories are fixed-size tensors with a validity count, so shapes never change (no recompiles).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RMSNorm, MLP


@dataclass
class MemConfig:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 4
    d_model: int = 256
    mlp_mult: float = 4.0
    act: str = "relu2"
    seg: int = 256  # training segment length
    mem_len: int = 256  # XL memory per layer (0 = none)
    ptr_len: int = 2048  # pointer memory (0 = no pointer head)
    ptr_dim: int = 64
    rope_base: float = 10000.0
    rope_frac: float = 0.5
    max_pos: int = 4096
    key_offset: bool = True
    attn_gate: bool = True
    unet: bool = True
    x0_mix: bool = True
    smear: bool = True
    value_residual: bool = True
    softcap: float = 15.0
    resid_dropout: float = 0.0
    type_dropout: float = 0.0
    ptr_aux: float = 0.0  # extra weight on the plain softmax CE (keeps the vocab head sharp)


class RotaryPos(nn.Module):
    def __init__(self, rot_dim, base, max_len):
        super().__init__()
        self.rot_dim = rot_dim
        inv = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        f = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", f.cos()[None, None], persistent=False)
        self.register_buffer("sin", f.sin()[None, None], persistent=False)

    def forward(self, x, start):  # x: B,H,T,D ; positions start..start+T-1
        T, r = x.size(2), self.rot_dim
        c, s = self.cos[:, :, start:start + T], self.sin[:, :, start:start + T]
        xr, xp = x[..., :r], x[..., r:]
        x1, x2 = xr.chunk(2, dim=-1)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c, xp], dim=-1)


class MemAttention(nn.Module):
    def __init__(self, cfg: MemConfig, i: int):
        super().__init__()
        self.cfg, self.i = cfg, i
        self.h, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rot_dim = int(self.hd * cfg.rope_frac) // 2 * 2
        self.rope = RotaryPos(self.rot_dim, cfg.rope_base, cfg.max_pos)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        if cfg.value_residual and i > 0:
            self.vlam = nn.Parameter(torch.tensor(0.5))
        if cfg.attn_gate:
            self.gate = nn.Linear(cfg.d_model, cfg.n_head, bias=True)

    def forward(self, x, v1, mk, mv, mem_ok):
        """mk, mv: (B,H,M,hd) raw (un-rotated) memory keys/values or None; mem_ok: (M,) bool."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        if self.cfg.value_residual:
            if self.i == 0:
                v1 = v
            else:
                v = torch.lerp(v, v1, self.vlam)
        M = 0 if mk is None else mk.size(2)
        kc = k if M == 0 else torch.cat([mk, k], dim=2)
        vc = v if M == 0 else torch.cat([mv, v], dim=2)
        S = M + T
        qr = self.rope(q, M)
        kr = self.rope(kc, 0)
        if self.cfg.key_offset:  # non-rotary half of each key carries the previous token's features
            o = self.rot_dim
            kr = torch.cat([kr[..., :o], F.pad(kr[:, :, :-1, o:], (0, 0, 1, 0))], dim=-1)
        if M == 0:
            y = F.scaled_dot_product_attention(qr, kr, vc, is_causal=True)
        else:
            j = torch.arange(S)
            mem_part = torch.cat([mem_ok, torch.zeros(T, dtype=torch.bool)])[None, :]
            allowed = mem_part | ((j[None, :] - M <= torch.arange(T)[:, None]) & (j[None, :] >= M))
            mask = torch.zeros(T, S).masked_fill(~allowed, float("-inf"))
            y = F.scaled_dot_product_attention(qr, kr, vc, attn_mask=mask)
        if self.cfg.attn_gate:
            y = y * torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y), v1, k, v


class MemBlock(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.cfg = cfg
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = MemAttention(cfg, i)
        self.mlp = MLP(cfg)
        if cfg.x0_mix:
            self.lam = nn.Parameter(torch.tensor([1.0, 0.0]))

    def _drop(self, y):
        if self.training and self.cfg.resid_dropout > 0:
            y = F.dropout(y, self.cfg.resid_dropout, True)
        return y

    def forward(self, x, x0, v1, mk, mv, mem_ok):
        if self.cfg.x0_mix:
            x = self.lam[0] * x + self.lam[1] * x0
        a, v1, k, v = self.attn(self.n1(x), v1, mk, mv, mem_ok)
        x = x + self._drop(a)
        x = x + self._drop(self.mlp(self.n2(x)))
        return x, v1, k, v


class PointerXL(nn.Module):
    def __init__(self, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.wte = nn.Embedding(cfg.vocab_size, d)
        self.blocks = nn.ModuleList([MemBlock(cfg, i) for i in range(cfg.n_layer)])
        self.nf = RMSNorm(d)
        self.logit_temp = nn.Parameter(torch.zeros(()))
        if cfg.unet:
            self.skip_w = nn.Parameter(torch.ones(cfg.n_layer // 2))
        if cfg.smear:
            self.smear_gate = nn.Linear(12, 1, bias=True)
            self.smear_lam = nn.Parameter(torch.zeros(()))
        if cfg.ptr_len:
            self.ptr_q = nn.Linear(d, cfg.ptr_dim, bias=False)
            self.ptr_k = nn.Linear(d, cfg.ptr_dim, bias=False)
            self.ptr_sent = nn.Linear(d, 1, bias=True)
            self.ptr_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.wte.weight, std=1.0 / math.sqrt(d))
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp.proj.weight)
            if cfg.attn_gate:
                nn.init.zeros_(b.attn.gate.weight)
                nn.init.constant_(b.attn.gate.bias, 3.0)
        if cfg.ptr_len:  # start mostly on the vocabulary softmax: sigmoid-ish sentinel preference
            nn.init.constant_(self.ptr_sent.bias, 3.0)

    # ---------------------------------------------------------------- state
    def init_state(self, B, mem_len=None, ptr_len=None):
        c = self.cfg
        H, hd = c.n_head, c.d_model // c.n_head
        M = c.mem_len if mem_len is None else mem_len
        P = c.ptr_len if ptr_len is None else ptr_len
        st = {"n_mem": torch.zeros((), dtype=torch.long), "n_ptr": torch.zeros((), dtype=torch.long)}
        st["mk"] = [torch.zeros(B, H, M, hd) for _ in range(c.n_layer)] if M else None
        st["mv"] = [torch.zeros(B, H, M, hd) for _ in range(c.n_layer)] if M else None
        st["pk"] = torch.zeros(B, P, c.ptr_dim) if P else None
        st["py"] = torch.zeros(B, P, dtype=torch.long) if P else None
        return st

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.wte.weight.numel() if non_embedding else n

    # ---------------------------------------------------------------- forward
    def forward(self, idx, y, mk, mv, n_mem, pk, py, n_ptr, reduction="mean"):
        """One segment. Returns (loss or per-token nll, new_mk, new_mv, new_pk, new_py).
        Memories are fixed-size; n_mem / n_ptr (0-dim long tensors) count valid trailing slots."""
        c = self.cfg
        B, T = idx.shape
        x = F.rms_norm(self.wte(idx), (c.d_model,))
        if self.training and c.type_dropout > 0:
            keep = (torch.rand(c.vocab_size) >= c.type_dropout).to(x.dtype)
            x = x * (keep[idx] / (1 - c.type_dropout)).unsqueeze(-1)
        if c.smear:
            g = torch.sigmoid(self.smear_gate(x[:, 1:, :12]))
            x = torch.cat([x[:, :1], x[:, 1:] + self.smear_lam * g * x[:, :-1]], dim=1)
        x0 = x
        M = mk[0].size(2) if mk is not None else 0
        mem_ok = torch.arange(M) >= (M - n_mem) if M else None
        v1, skips, new_k, new_v = None, [], [], []
        L = c.n_layer
        for i, b in enumerate(self.blocks):
            if c.unet and i >= L - L // 2:
                x = x + self.skip_w[L - 1 - i] * skips.pop()
            x, v1, k, v = b(x, x0, v1, mk[i] if M else None, mv[i] if M else None, mem_ok)
            if c.unet and i < L // 2:
                skips.append(x)
            if M:
                new_k.append(torch.cat([mk[i], k.detach()], dim=2)[:, :, -M:])
                new_v.append(torch.cat([mv[i], v.detach()], dim=2)[:, :, -M:])
        h = self.nf(x)
        z = F.linear(h * (self.logit_temp.exp() / math.sqrt(c.d_model)), self.wte.weight)
        if c.softcap > 0:
            z = c.softcap * torch.tanh(z / c.softcap)
        nll_v = F.cross_entropy(z.float().reshape(-1, z.size(-1)), y.reshape(-1), reduction="none").view(B, T)
        if pk is None:
            nll = nll_v
            new_pk, new_py = None, None
        else:
            P = pk.size(1)
            qp = self.ptr_q(h)
            kp = self.ptr_k(h)
            Kc = torch.cat([pk, kp], dim=1)  # B, P+T, dp
            Yc = torch.cat([py, y], dim=1)  # B, P+T (value = next token of each position)
            s = (qp @ Kc.transpose(1, 2)) * (self.ptr_scale.exp() / math.sqrt(c.ptr_dim))  # B,T,P+T
            j = torch.arange(P + T)
            ok = ((j[None, :] < P) & (j[None, :] >= P - n_ptr)) | ((j[None, :] >= P) & (j[None, :] - P < torch.arange(T)[:, None]))
            s = s.masked_fill(~ok[None], float("-inf"))
            sent = self.ptr_sent(h)  # B,T,1
            la = torch.log_softmax(torch.cat([s, sent], dim=-1).float(), dim=-1)
            match = (Yc[:, None, :] == y[:, :, None]) & ok[None]
            lp_ptr = torch.logsumexp(la[..., :-1].masked_fill(~match, float("-inf")), dim=-1)
            nll = -torch.logaddexp(la[..., -1] - nll_v, lp_ptr)
            new_pk = Kc.detach()[:, -P:]
            new_py = Yc[:, -P:]
        if reduction == "none":
            return nll, new_k, new_v, new_pk, new_py
        loss = nll.mean()
        if pk is not None and c.ptr_aux > 0:
            loss = loss + c.ptr_aux * nll_v.mean()
        return loss, new_k, new_v, new_pk, new_py

    @staticmethod
    def advance(st, out, seg):
        """Carry memories forward after a segment of `seg` tokens."""
        _, nk, nv, npk, npy = out
        if st["mk"] is not None:
            st["mk"], st["mv"] = nk, nv
            st["n_mem"] = torch.clamp(st["n_mem"] + seg, max=st["mk"][0].size(2))
        if st["pk"] is not None:
            st["pk"], st["py"] = npk, npy
            st["n_ptr"] = torch.clamp(st["n_ptr"] + seg, max=st["pk"].size(1))
        return st

    @staticmethod
    def reset(st):
        st["n_mem"] = torch.zeros((), dtype=torch.long)
        st["n_ptr"] = torch.zeros((), dtype=torch.long)
        return st
