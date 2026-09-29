"""Compact, CPU-friendly causal Transformer LM with switchable modern components.

All extras default to off so older checkpoints load unchanged.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 256
    mlp_mult: float = 4.0
    act: str = "relu2"  # relu2 | lrelu2 | swiglu | gelu
    ctx: int = 256
    tie: bool = True
    qk_norm: bool = True
    rope_base: float = 10000.0
    rope_frac: float = 1.0  # fraction of head dims that get rotary (partial RoPE)
    emb_dropout: float = 0.0  # dropout on embedding output
    resid_dropout: float = 0.0  # dropout on each sublayer output before residual add
    attn_dropout: float = 0.0
    word_dropout: float = 0.0  # drop whole token embeddings (AWD-style embedding dropout, per token)
    value_residual: bool = False  # ResFormer: mix first-layer values into every layer
    softcap: float = 0.0  # logit soft-capping (0 = off)
    norm_out: bool = True
    zero_init_proj: bool = True
    stoch_depth: float = 0.0  # max layer drop prob (linearly increasing with depth)
    attn_gate: bool = False  # per-head sigmoid output gate (Gated Attention, Qiu et al. 2025)
    key_offset: bool = False  # shift the non-rotary half of each key back one position (1-layer induction)
    unet: bool = False  # U-Net skips: layer i output feeds layer L-1-i (learnable weights)
    x0_mix: bool = False  # re-inject token embedding x0 at every block with learnable lambdas
    smear: bool = False  # SmearGate: x_t += sigmoid(g(x_t)) * lam * x_{t-1} on the embedding
    canon: bool = False  # Canon layers: causal depthwise conv (k=4) residual before attn and mlp
    emb_norm: bool = False  # RMS-normalise the input embedding (decouples input scale from tied-row norms)
    type_dropout: float = 0.0  # AWD-LSTM embedding dropout: drop whole vocabulary rows (input side) per batch


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if affine else None

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class Rotary(nn.Module):
    def __init__(self, dim, rot_dim, base, max_len):
        super().__init__()
        self.rot_dim = rot_dim
        inv = 1.0 / (base ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        t = torch.arange(max_len).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos()[None, None], persistent=False)
        self.register_buffer("sin", f.sin()[None, None], persistent=False)

    def forward(self, x):  # x: B,H,T,D
        T = x.size(2)
        c, s = self.cos[:, :, :T], self.sin[:, :, :T]
        r = self.rot_dim
        xr, xp = (x, None) if r == x.size(-1) else (x[..., :r], x[..., r:])
        x1, x2 = xr.chunk(2, dim=-1)
        y = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        return y if xp is None else torch.cat([y, xp], dim=-1)


class CausalDWConv(nn.Module):
    """Canon layer: y = x + sum_k w_k * x_{t-k}, k=0..K-1 (depthwise, causal, zero-init)."""

    def __init__(self, d, K=4):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.zeros(K, d))

    def forward(self, x):  # B,T,C
        y = x + x * self.w[0]
        for k in range(1, self.K):
            y = y + F.pad(x[:, :-k], (0, 0, k, 0)) * self.w[k]
        return y


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        rot = int(self.hd * cfg.rope_frac) // 2 * 2
        self.rot_dim = rot
        self.rot = Rotary(self.hd, rot, cfg.rope_base, max(cfg.ctx, 4096))
        if cfg.qk_norm:
            self.qn = RMSNorm(self.hd)
            self.kn = RMSNorm(self.hd)
        self.layer_idx = layer_idx
        if cfg.value_residual and layer_idx > 0:
            self.vlam = nn.Parameter(torch.tensor(0.5))
        if cfg.attn_gate:
            self.gate = nn.Linear(cfg.d_model, cfg.n_head, bias=True)

    def forward(self, x, v1=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.cfg.qk_norm:
            q, k = self.qn(q), self.kn(k)
        q, k = self.rot(q), self.rot(k)
        if self.cfg.key_offset:
            o = self.rot_dim if self.rot_dim < self.hd else self.hd // 2
            ks = F.pad(k[:, :, :-1, o:], (0, 0, 1, 0))
            k = torch.cat([k[..., :o], ks], dim=-1)
        if self.cfg.value_residual:
            if self.layer_idx == 0:
                v1 = v
            else:
                v = (1 - self.vlam) * v + self.vlam * v1
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.cfg.attn_dropout if self.training else 0.0
        )  # B,H,T,D
        if self.cfg.attn_gate:
            g = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)  # B,H,T,1
            y = y * g
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y), v1


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.act = cfg.act
        hid = int(cfg.mlp_mult * cfg.d_model)
        if cfg.act == "swiglu":
            hid = int(2 * hid / 3)
            hid = (hid + 31) // 32 * 32
            self.fc = nn.Linear(cfg.d_model, 2 * hid, bias=False)
        else:
            self.fc = nn.Linear(cfg.d_model, hid, bias=False)
        self.proj = nn.Linear(hid, cfg.d_model, bias=False)

    def forward(self, x):
        h = self.fc(x)
        if self.act == "relu2":
            h = F.relu(h).square()
        elif self.act == "lrelu2":
            h = F.leaky_relu(h, 0.5).square()
        elif self.act == "swiglu":
            a, b = h.chunk(2, dim=-1)
            h = F.silu(a) * b
        else:
            h = F.gelu(h)
        return self.proj(h)


class Block(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.cfg = cfg
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, i)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.rd = cfg.resid_dropout
        self.sd = cfg.stoch_depth * i / max(1, cfg.n_layer - 1)
        if cfg.x0_mix:
            self.lam = nn.Parameter(torch.tensor([1.0, 0.0]))
        if cfg.canon:
            self.cA = CausalDWConv(cfg.d_model)
            self.cC = CausalDWConv(cfg.d_model)

    def _drop(self, y):
        if self.training and self.rd > 0:
            y = F.dropout(y, self.rd, True)
        return y

    def forward(self, x, v1=None, x0=None):
        if self.cfg.x0_mix:
            x = self.lam[0] * x + self.lam[1] * x0
        if self.training and self.sd > 0:
            keep = (torch.rand(x.size(0), 1, 1, device=x.device) >= self.sd).to(x.dtype) / (1 - self.sd)
        else:
            keep = None
        h = self.n1(x)
        if self.cfg.canon:
            h = self.cA(h)
        a, v1 = self.attn(h, v1)
        a = self._drop(a)
        if keep is not None:
            a = a * keep
        x = x + a
        h = self.n2(x)
        if self.cfg.canon:
            h = self.cC(h)
        m = self._drop(self.mlp(h))
        if keep is not None:
            m = m * keep
        return x + m, v1


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model) if cfg.norm_out else nn.Identity()
        if not cfg.tie:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            # tied: input = E[x]*sqrt(d) (RMS 1); logits = <h, E[v]>/sqrt(d) * exp(logit_temp).
            # Without the 1/sqrt(d) the zero-init residual net starts as an identity map that
            # predicts the *current* token with logit sqrt(d) (initial loss ~16 instead of ln V).
            self.logit_temp = nn.Parameter(torch.zeros(()))
        if cfg.unet:
            self.skip_w = nn.Parameter(torch.ones(cfg.n_layer // 2))
        if cfg.smear:
            self.smear_gate = nn.Linear(12, 1, bias=True)
            self.smear_lam = nn.Parameter(torch.zeros(()))
        self.apply(self._init)
        if cfg.zero_init_proj:
            for b in self.blocks:
                nn.init.zeros_(b.attn.proj.weight)
                nn.init.zeros_(b.mlp.proj.weight)
        for b in self.blocks:
            if cfg.attn_gate:  # sigmoid(0+3)~0.95: starts close to ungated attention
                nn.init.zeros_(b.attn.gate.weight)
                nn.init.constant_(b.attn.gate.bias, 3.0)
        if not cfg.tie:
            nn.init.zeros_(self.head.weight)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=1.0 if not self.cfg.tie else 1.0 / math.sqrt(self.cfg.d_model))

    def head_weight(self):
        return self.wte.weight if self.cfg.tie else self.head.weight

    def hidden(self, idx):
        cfg = self.cfg
        x = self.wte(idx)
        if cfg.emb_norm:
            x = F.rms_norm(x, (x.size(-1),))
        elif cfg.tie:
            x = x * math.sqrt(cfg.d_model)
        if self.training and cfg.type_dropout > 0:
            keep = (torch.rand(cfg.vocab_size, device=idx.device) >= cfg.type_dropout).to(x.dtype)
            x = x * (keep[idx] / (1 - cfg.type_dropout)).unsqueeze(-1)
        if self.training and cfg.word_dropout > 0:
            m = (torch.rand(idx.shape, device=idx.device) >= cfg.word_dropout).to(x.dtype)
            x = x * m.unsqueeze(-1) / (1 - cfg.word_dropout)
        if cfg.smear:
            g = torch.sigmoid(self.smear_gate(x[:, 1:, :12]))
            x = torch.cat([x[:, :1], x[:, 1:] + self.smear_lam * g * x[:, :-1]], dim=1)
        if self.training and cfg.emb_dropout > 0:
            x = F.dropout(x, cfg.emb_dropout, True)
        x0 = x
        v1 = None
        L = cfg.n_layer
        skips = []
        for i, b in enumerate(self.blocks):
            if cfg.unet and i >= L - L // 2:
                x = x + self.skip_w[L - 1 - i] * skips.pop()
            x, v1 = b(x, v1, x0)
            if cfg.unet and i < L // 2:
                skips.append(x)
        return self.nf(x)

    def logits(self, h):
        if self.cfg.tie:
            h = h * (self.logit_temp.exp() / math.sqrt(self.cfg.d_model))
        z = F.linear(h, self.head_weight())
        if self.cfg.softcap > 0:
            z = self.cfg.softcap * torch.tanh(z / self.cfg.softcap)
        return z

    def forward(self, idx, targets=None, reduction="mean"):
        h = self.hidden(idx)
        z = self.logits(h)
        if targets is None:
            return z
        return F.cross_entropy(z.float().view(-1, z.size(-1)), targets.reshape(-1), reduction=reduction)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n
