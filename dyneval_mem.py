"""Strictly-causal streaming dynamic evaluation for PointerXL.

One pass over the split in file order (single stream). For each segment of `seg` targets:
  1. SCORE with the current weights, attention memory and pointer memory (only past tokens);
  2. ADAPT: one optimizer step on that segment's (already scored) NLL, optionally decaying the
     weights toward the trained weights theta0.
Memories carry forward (detached). Each target is scored exactly once, before any update that
used it, so sum NLL / ln2 / bytes is a valid BPB.
"""
import argparse, json, math, os, time

import numpy as np
import torch

from memlm import PointerXL, MemConfig


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PointerXL(MemConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m, ck


def split_stream(data, split, eos):
    arr = np.load(os.path.join(data, f"{split}.npy")).astype(np.int64)
    return torch.cat([torch.tensor([eos]), torch.from_numpy(arr)])


def token_bytes(data):
    """Exact byte length of every token id: in byte-level BPE each character of a token string
    stands for exactly one byte (GPT-2 bytes_to_unicode map)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(data, "tokenizer.json"))
    return torch.tensor([0 if tok.id_to_token(i) == "<|eos|>" else len(tok.id_to_token(i))
                         for i in range(tok.get_vocab_size())])


def select(model, which):
    named = list(model.named_parameters())
    if which == "all":
        return named
    if which == "noemb":
        return [(n, p) for n, p in named if "wte" not in n]
    if which == "mats":
        return [(n, p) for n, p in named if p.ndim == 2]
    raise ValueError(which)


def grad_rms(model, train, seg, mem_len, ptr_len, n_batches, named, seed=0):
    """RMS of per-segment gradients measured by streaming over random TRAIN windows."""
    g = torch.Generator().manual_seed(seed)
    ms = [torch.zeros_like(p) for _, p in named]
    ps = [p for _, p in named]
    model.eval()
    warm = 8  # segments of warm-up per window so memories are populated
    cnt = 0
    for _ in range(max(1, n_batches // 8)):
        s0 = int(torch.randint(0, len(train) - seg * (warm + 8) - 1, (1,), generator=g))
        st = model.init_state(1, mem_len, ptr_len)
        for i in range(warm + 8):
            x = train[s0 + i * seg: s0 + (i + 1) * seg][None]
            y = train[s0 + i * seg + 1: s0 + (i + 1) * seg + 1][None]
            if i < warm:
                with torch.no_grad():
                    out = model(x, y, st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none")
            else:
                out = model(x, y, st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none")
                grads = torch.autograd.grad(out[0].mean(), ps, allow_unused=True)
                for m_, gr in zip(ms, grads):
                    if gr is not None:
                        m_ += gr * gr
                cnt += 1
            st = model.advance(st, out, seg)
    return [(m_ / cnt).sqrt() for m_ in ms]


def dyn_eval(model, stream, seg, mem_len, ptr_len, lr=0.0, opt="adam", decay=0.0, beta1=0.9, beta2=0.999,
             params="all", rms=None, eps=1e-3, emb_lr_mult=1.0, max_targets=0, collect=None):
    model.eval()
    n = len(stream) - 1 if not max_targets else min(max_targets, len(stream) - 1)
    named = select(model, params)
    ps = [p for _, p in named]
    mult = [emb_lr_mult if "wte" in nm else 1.0 for nm, _ in named]
    adapt = lr > 0
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(adapt)
    theta0 = [p.detach().clone() for p in ps] if adapt else None
    if adapt and opt == "adam":
        groups = {}
        for p, mm in zip(ps, mult):
            groups.setdefault(mm, []).append(p)
        o = torch.optim.Adam([dict(params=v, lr=lr * mm) for mm, v in groups.items()], betas=(beta1, beta2))
        if rms is not None:  # warm-start second moments from TRAIN gradient statistics
            s_ = float(int(1 / (1 - beta2)))  # bias-corrected so that v_hat = rms^2 exactly
            for p, r in zip(ps, rms):
                o.state[p]["step"] = torch.tensor(s_)
                o.state[p]["exp_avg"] = torch.zeros_like(p)
                o.state[p]["exp_avg_sq"] = (r * r * (1 - beta2 ** s_)).clone()
    if adapt and opt == "rms":
        mean_rms = torch.stack([r.mean() for r in rms]).mean()
        denom = [r + eps * mean_rms for r in rms]
        decrate = [(r / mean_rms).clamp(max=1.0 / decay) for r in rms] if decay > 0 else None
    nseg = -(-n // seg)
    pad = torch.full((nseg * seg + 1,), int(stream[0]), dtype=torch.long)
    pad[: n + 1] = stream[: n + 1]
    st = model.init_state(1, mem_len, ptr_len)
    total, count = 0.0, 0
    for i in range(nseg):
        x = pad[i * seg:(i + 1) * seg][None]
        y = pad[i * seg + 1:(i + 1) * seg + 1][None]
        L = min(seg, n - i * seg)
        with torch.set_grad_enabled(adapt):
            out = model(x, y, st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none")
        nll = out[0][0, :L]
        sc = nll.detach().double()
        total += sc.sum().item()
        count += L
        if collect is not None:
            collect.append(sc.clone())
        if adapt:
            nll.mean().backward()
            with torch.no_grad():
                if opt == "adam":
                    o.step()
                    o.zero_grad(set_to_none=True)
                    if decay > 0:
                        torch._foreach_lerp_(ps, theta0, decay)
                elif opt == "rms":
                    for j, p in enumerate(ps):
                        if p.grad is None:
                            continue
                        p.add_(p.grad / denom[j], alpha=-lr * mult[j])
                        if decay > 0:
                            p.add_(decay * decrate[j] * (theta0[j] - p))
                        p.grad = None
                else:  # sgd
                    for j, p in enumerate(ps):
                        if p.grad is not None:
                            p.add_(p.grad, alpha=-lr * mult[j])
                            if decay > 0:
                                p.lerp_(theta0[j], decay)
                            p.grad = None
        st = model.advance(st, out, seg)
    assert count == n
    if adapt:
        with torch.no_grad():
            for p, p0 in zip(ps, theta0):
                p.copy_(p0)
    for p in model.parameters():
        p.requires_grad_(True)
    return total, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--mem_len", type=int, default=-1, help="-1: 2*train_seg - seg (keeps trained key range)")
    ap.add_argument("--ptr_len", type=int, default=-1, help="-1: as trained")
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--opt", default="adam", choices=["adam", "rms", "sgd"])
    ap.add_argument("--decay", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--params", default="all")
    ap.add_argument("--stat_batches", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--emb_lr_mult", type=float, default=1.0)
    ap.add_argument("--max_targets", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    model, ck = load_model(a.ckpt)
    cfg = ck["cfg"]
    data = ck["args"]["data"]
    meta = json.load(open(os.path.join(data, "meta.json")))
    stream = split_stream(data, a.split, meta["eos_id"])
    mem_len = (cfg["mem_len"] + cfg["seg"] - a.seg) if a.mem_len < 0 else a.mem_len
    if cfg["mem_len"] == 0:
        mem_len = 0
    ptr_len = cfg["ptr_len"] if a.ptr_len < 0 else a.ptr_len
    if cfg["ptr_len"] == 0:
        ptr_len = 0
    rms = None
    if a.stat_batches:
        train = torch.from_numpy(np.load(os.path.join(data, "train.npy")).astype(np.int64))
        rms = grad_rms(model, train, a.seg, mem_len, ptr_len, a.stat_batches, select(model, a.params))
    t = time.time()
    col = [] if a.save else None
    nll, n = dyn_eval(model, stream, a.seg, mem_len, ptr_len, a.lr, a.opt, a.decay, a.beta1, a.beta2,
                      a.params, rms, a.eps, a.emb_lr_mult, a.max_targets, col)
    if a.save:
        assert not a.max_targets, "--save requires a full-split run"
        np.save(a.save, torch.cat(col).numpy())
    nbytes = meta[a.split]["bytes"]
    if n < meta[a.split]["tokens"]:  # exact byte count of the scored prefix
        nbytes = int(token_bytes(data)[stream[1:n + 1]].sum())
    print("RESULT", json.dumps({"split": a.split, "bpb": nll / math.log(2) / nbytes, "loss": nll / n, "n": n,
                                "seg": a.seg, "mem_len": mem_len, "ptr_len": ptr_len, "lr": a.lr, "opt": a.opt,
                                "decay": a.decay, "params": a.params, "beta1": a.beta1, "beta2": a.beta2,
                                "stat": a.stat_batches, "emb_lr_mult": a.emb_lr_mult,
                                "time": time.time() - t}), flush=True)


if __name__ == "__main__":
    main()
