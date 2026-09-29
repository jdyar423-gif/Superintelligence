"""Strictly-causal dynamic evaluation (test-time training) + neural cache for the WT2 LM.

The eval stream (<eos> + split tokens) is processed left to right in segments of `seg` targets.
For each segment:
  1. SCORE: forward the last `ctx` tokens of history ending at the segment and record the NLL
     of the `seg` new targets with the *current* weights (optionally mixed with a neural cache
     over already-scored positions).
  2. ADAPT: backprop the model NLL of that same segment and take one update step, optionally
     decaying the weights back toward the trained weights theta0.
Every token is scored before any gradient containing it is applied, so the procedure defines a
valid causal distribution over the whole split; BPB = sum NLL / ln2 / bytes.

Update rules:
  adam : torch Adam (optionally warm-started second moments from TRAIN gradient statistics)
  rms  : Krause et al. (2018/2019) RMS dynamic eval: p += -lr*g/(RMS+eps) + lam*decrate*(p0-p),
         RMS = sqrt(mean g^2) measured on TRAIN segments, decrate = RMS/mean(RMS) clipped at 1/lam.
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m, ck


def stream_for(data_dir, split, eos):
    arr = np.load(os.path.join(data_dir, f"{split}.npy")).astype(np.int64)
    return torch.cat([torch.tensor([eos]), torch.from_numpy(arr)])


def select_params(model, which):
    named = list(model.named_parameters())
    if which == "all":
        return named
    if which == "noemb":
        return [(n, p) for n, p in named if "wte" not in n and "head" not in n]
    if which == "mats":
        return [(n, p) for n, p in named if p.ndim == 2]
    raise ValueError(which)


def grad_stats(model, train, ctx, seg, n_batches, params, seed=0):
    """Root-mean-square gradient of the per-segment mean NLL on random TRAIN windows."""
    g = torch.Generator().manual_seed(seed)
    ms = [torch.zeros_like(p) for _, p in params]
    model.train(False)
    for _ in range(n_batches):
        s = int(torch.randint(0, len(train) - ctx - 1, (1,), generator=g))
        x = train[s:s + ctx][None]
        y = train[s + 1:s + ctx + 1][None]
        h = model.hidden(x)[:, -seg:]
        z = model.logits(h).float()
        loss = F.cross_entropy(z.view(-1, z.size(-1)), y[:, -seg:].reshape(-1))
        grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
        for m_, gr in zip(ms, grads):
            if gr is not None:
                m_ += gr * gr
    return [(m_ / n_batches).sqrt() for m_ in ms]


def dynamic_eval(model, stream, ctx, seg, lr, opt="adam", decay=0.0, beta1=0.9, beta2=0.999,
                 params="all", max_targets=0, log_every=0, cache=None, grad_clip=0.0, rms=None,
                 eps=1e-3, emb_lr_mult=1.0, temp=1.0, collect=None):
    """Returns (sum_nll_nats, n_targets). lr=0 => static sliding-window eval with stride=seg.
    `rms`: list of RMS gradient tensors (train statistics) aligned with the selected params.
    `collect`: optional list; per-token model log-probs of targets get appended (for mixing)."""
    model.train(False)
    n_tgt = len(stream) - 1 if not max_targets else min(max_targets, len(stream) - 1)
    named = select_params(model, params)
    ps = [p for _, p in named]
    lr_mult = [emb_lr_mult if ("wte" in n or "head" in n) else 1.0 for n, _ in named]
    adapt = lr > 0
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(adapt)
    theta0 = [p.detach().clone() for p in ps] if adapt else None
    if adapt and opt == "adam":
        groups = {}
        for p, m in zip(ps, lr_mult):
            groups.setdefault(m, []).append(p)
        o = torch.optim.Adam([dict(params=v, lr=lr * m) for m, v in groups.items()],
                             betas=(beta1, beta2), eps=1e-8)
        if rms is not None:  # warm-start second moments from train statistics
            for p, r in zip(ps, rms):
                o.state[p]["step"] = torch.tensor(float(int(1 / (1 - beta2))))
                o.state[p]["exp_avg"] = torch.zeros_like(p)
                o.state[p]["exp_avg_sq"] = (r * r).clone()
    if adapt and opt == "rms":
        assert rms is not None
        mean_rms = torch.stack([r.mean() for r in rms]).mean()
        eps_abs = eps * mean_rms
        denom = [r + eps_abs for r in rms]
        if decay > 0:
            decrate = [(r / mean_rms).clamp(max=1.0 / decay) for r in rms]
    total, count = 0.0, 0
    t0 = time.time()
    use_cache = cache is not None and cache.get("lam", 0) > 0
    if use_cache:
        cK = torch.empty(0, model.cfg.d_model)
        cY = torch.empty(0, dtype=torch.long)
    pos = 0  # targets scored so far; the next targets are stream[pos+1 : pos+1+seg]
    while pos < n_tgt:
        L = min(seg, n_tgt - pos)
        end = pos + L
        start = max(0, end - ctx)
        x = stream[start:end][None]
        y = stream[start + 1:end + 1][None]
        yt = y[0, -L:]
        with torch.set_grad_enabled(adapt):
            h = model.hidden(x)[:, -L:]
            z = model.logits(h).float().view(L, -1)
            if temp != 1.0:
                z = z / temp
            logp = torch.log_softmax(z, -1)
            nll = -logp.gather(1, yt[:, None]).squeeze(1)
        with torch.no_grad():
            if use_cache:
                hs = h[0].detach()
                hs = hs / hs.norm(dim=-1, keepdim=True)
                K = torch.cat([cK, hs])
                Y = torch.cat([cY, yt])
                M = cK.size(0)
                s = cache["theta"] * (hs @ K.t())  # L x (M+L)
                # position j may use history + in-segment positions < j (their targets are known inputs)
                mask = torch.ones(L, M + L, dtype=torch.bool)
                mask[:, M:] = torch.tril(torch.ones(L, L, dtype=torch.bool), -1)
                s = s.masked_fill(~mask, float("-inf"))
                has = mask.any(1)
                att = torch.softmax(s, dim=-1).nan_to_num(0.0)
                pc = torch.zeros(L, z.size(-1)).scatter_add_(1, Y[None].expand(L, -1), att)
                lam = cache["lam"] * has.float()
                p_t = (1 - lam) * nll.detach().neg().exp() + lam * pc.gather(1, yt[:, None]).squeeze(1)
                score = -torch.log(p_t)
                cK = K[-cache["size"]:]
                cY = Y[-cache["size"]:]
            else:
                score = nll.detach()
            total += score.sum().item()
            if collect is not None:
                collect.append(score.clone())
        count += L
        if adapt:
            nll.mean().backward()
            with torch.no_grad():
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ps, grad_clip)
                if opt == "adam":
                    o.step()
                    o.zero_grad(set_to_none=True)
                    if decay > 0:
                        torch._foreach_lerp_(ps, theta0, decay)
                elif opt == "rms":
                    for i, p in enumerate(ps):
                        if p.grad is None:
                            continue
                        p.add_(p.grad / denom[i], alpha=-lr * lr_mult[i])
                        if decay > 0:
                            p.add_(decay * decrate[i] * (theta0[i] - p))
                        p.grad = None
                elif opt == "sgd":
                    for i, p in enumerate(ps):
                        if p.grad is not None:
                            p.add_(p.grad, alpha=-lr * lr_mult[i])
                            if decay > 0:
                                p.lerp_(theta0[i], decay)
                            p.grad = None
        pos = end
        if log_every and (pos // seg) % log_every == 0:
            print(f"  {pos}/{n_tgt} nll {total/count:.4f} {time.time()-t0:.0f}s", flush=True)
    if adapt:  # restore trained weights so the model object can be reused
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
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--seg", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--opt", default="adam", choices=["adam", "rms", "sgd"])
    ap.add_argument("--decay", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--params", default="all")
    ap.add_argument("--max_targets", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=0.0)
    ap.add_argument("--stat_batches", type=int, default=0, help="train segments for RMS grad stats (0=off)")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--emb_lr_mult", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--cache_lam", type=float, default=0.0)
    ap.add_argument("--cache_theta", type=float, default=20.0)
    ap.add_argument("--cache_size", type=int, default=4096)
    ap.add_argument("--save", default="", help="save per-token NLL (nats) to this .npy")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    model, ck = load_model(a.ckpt)
    data = ck["args"]["data"]
    meta = json.load(open(os.path.join(data, "meta.json")))
    stream = stream_for(data, a.split, meta["eos_id"])
    ctx = a.ctx or ck["cfg"]["ctx"]
    rms = None
    if a.stat_batches:
        train = torch.from_numpy(np.load(os.path.join(data, "train.npy")).astype(np.int64))
        rms = grad_stats(model, train, ctx, a.seg, a.stat_batches, select_params(model, a.params))
    t = time.time()
    col = [] if a.save else None
    nll, n = dynamic_eval(model, stream, ctx, a.seg, a.lr, a.opt, a.decay, a.beta1, a.beta2, a.params,
                          a.max_targets, log_every=0, grad_clip=a.grad_clip, rms=rms, eps=a.eps,
                          emb_lr_mult=a.emb_lr_mult, temp=a.temp, collect=col,
                          cache=dict(lam=a.cache_lam, theta=a.cache_theta, size=a.cache_size))
    if a.save:
        np.save(a.save, torch.cat(col).numpy().astype(np.float64))
    split_bytes = meta[a.split]["bytes"]
    if a.max_targets:
        split_bytes = split_bytes * n / meta[a.split]["tokens"]  # approx bytes for partial runs
    bpb = nll / math.log(2) / split_bytes
    print("RESULT", json.dumps({"split": a.split, "bpb": bpb, "loss": nll / n, "n": n, "lr": a.lr,
                                "seg": a.seg, "opt": a.opt, "decay": a.decay, "params": a.params,
                                "emb_lr_mult": a.emb_lr_mult, "temp": a.temp,
                                "cache": [a.cache_lam, a.cache_theta, a.cache_size],
                                "time": time.time() - t}), flush=True)


if __name__ == "__main__":
    main()
