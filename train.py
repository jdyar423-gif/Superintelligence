"""Train a subword causal LM on WikiText-2 on CPU and report validation/test bits-per-byte.

BPB = sum_t -log2 p(x_t | x_<t) over every test token / #UTF-8 bytes of the raw test split.
The eval stream is prefixed with one <|eos|> token as the initial context; every real token is scored.
"""
import argparse, json, math, os, time, copy
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from optim import Muon

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bpe8k")
    ap.add_argument("--out", default="runs/debug")
    ap.add_argument("--seed", type=int, default=1337)
    # model
    for f, v in asdict(GPTConfig()).items():
        if f == "vocab_size":
            continue
        t = type(v)
        if t is bool:
            ap.add_argument(f"--{f}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{f}", type=t, default=v)
    # optimisation
    ap.add_argument("--opt", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--lr", type=float, default=3e-3, help="AdamW lr (embeddings/scalars; all params if --opt adamw)")
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--mom_warmup", type=int, default=0, help="steps to warm Muon momentum 0.85->target")
    ap.add_argument("--cwd", type=int, default=0, help="cautious weight decay for Muon matrices")
    ap.add_argument("--polar", type=int, default=0, help="Polar Express NS coefficients")
    ap.add_argument("--normuon", type=int, default=0, help="NorMuon row normalisation")
    ap.add_argument("--ctx_short", type=int, default=0, help="train at this shorter ctx first (0=off)")
    ap.add_argument("--ctx_switch", type=float, default=0.7, help="progress fraction to switch to full ctx")
    ap.add_argument("--wd", type=float, default=0.1, help="decoupled weight decay for matrices")
    ap.add_argument("--emb_wd", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--time_budget", type=float, default=0.0, help="training seconds (overrides --steps)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--sched", default="cosine", choices=["cosine", "linear", "wsd"])
    ap.add_argument("--wsd_frac", type=float, default=0.3, help="fraction of steps in final decay for wsd")
    ap.add_argument("--min_lr_frac", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0=off)")
    ap.add_argument("--ema_start", type=float, default=0.5, help="fraction of training before EMA starts")
    ap.add_argument("--swa_start", type=float, default=0.0, help="uniform weight average from this fraction (0=off)")
    ap.add_argument("--swa_every", type=int, default=1)
    # eval
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_tokens", type=int, default=65536, help="val tokens for periodic eval (0=all)")
    ap.add_argument("--eval_stride", type=int, default=0, help="final eval stride (0 = ctx//2)")
    ap.add_argument("--final_test", type=int, default=1)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=25)
    return ap.parse_args(argv)


def load(path, split):
    return torch.from_numpy(np.load(os.path.join(path, f"{split}.npy")).astype(np.int64))


class EpochSampler:
    """Every epoch: random phase offset, non-overlapping windows, shuffled. Covers each token ~once/epoch."""

    def __init__(self, data, ctx, batch, gen):
        self.d, self.T, self.B, self.g = data, ctx, batch, gen
        self.queue = []
        self.epoch = 0

    def _refill(self):
        off = int(torch.randint(0, self.T, (1,), generator=self.g))
        starts = torch.arange(off, len(self.d) - self.T - 1, self.T)
        starts = starts[torch.randperm(len(starts), generator=self.g)]
        self.queue.extend(starts.tolist())
        self.epoch += 1

    def next(self):
        while len(self.queue) < self.B:
            self._refill()
        s = torch.tensor(self.queue[: self.B])
        self.queue = self.queue[self.B:]
        idx = s[:, None] + torch.arange(self.T + 1)[None]
        w = self.d[idx]
        return w[:, :-1], w[:, 1:]


@torch.no_grad()
def eval_nll(model, stream, ctx, stride, batch=16, max_tokens=0, hidden_fn=None):
    """Sliding-window NLL. stream[0] is the <eos> context token; targets are stream[1:].
    Every target is scored exactly once, with up to ctx-1 tokens of context. The LM head is
    only evaluated on rows that are scored. Returns (sum_nll_nats, n_targets)."""
    base = getattr(model, "_orig_mod", model)
    hidden_fn = hidden_fn or base.hidden
    was_training = base.training
    base.eval()
    if max_tokens:
        stream = stream[: max_tokens + 1]
    N = len(stream)
    n_tgt = N - 1
    ctx = min(ctx, n_tgt)
    starts = list(range(0, n_tgt - ctx + 1, stride))
    if starts[-1] != n_tgt - ctx:
        starts.append(n_tgt - ctx)
    scored_end = 0
    jobs = []
    for s in starts:
        lo = max(scored_end, s)  # target index range [lo, s+ctx)
        jobs.append((s, lo - s))
        scored_end = s + ctx
    total, count = 0.0, 0
    ar = torch.arange(ctx + 1)
    for i in range(0, len(jobs), batch):
        chunk = jobs[i: i + batch]
        st = [s for s, _ in chunk]
        st = st + [st[-1]] * (batch - len(st))  # pad to a fixed shape (no recompiles)
        w = stream[torch.tensor(st)[:, None] + ar[None]]
        x, y = w[:, :-1], w[:, 1:]
        h = hidden_fn(x)
        lo = min(off for _, off in chunk)
        z = base.logits(h[: len(chunk), lo:]).float()
        nll = F.cross_entropy(z.reshape(-1, z.size(-1)), y[: len(chunk), lo:].reshape(-1),
                              reduction="none").view(len(chunk), -1)
        for j, (_, off) in enumerate(chunk):
            total += nll[j, off - lo:].double().sum().item()
            count += ctx - off
    assert count == n_tgt, (count, n_tgt)
    base.train(was_training)
    return total, count


def lr_mult(step, frac, a):
    """Warmup in steps, then the schedule over training progress frac in [0,1]."""
    if step < a.warmup:
        return (step + 1) / a.warmup
    p = min(frac, 1.0)
    if a.sched == "cosine":
        m = 0.5 * (1 + math.cos(math.pi * p))
    elif a.sched == "linear":
        m = 1.0 - p
    else:  # wsd: constant, then linear decay to 0 over the final wsd_frac
        d0 = 1.0 - a.wsd_frac
        m = 1.0 if p < d0 else max(0.0, 1.0 - (p - d0) / a.wsd_frac)
    return a.min_lr_frac + (1 - a.min_lr_frac) * m


def build_optimizers(model, a):
    mats, embs, scal = [], [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "wte" not in n and "head" not in n and min(p.shape) >= 32 and "gate" not in n:
            mats.append(p)
            if n.endswith("attn.qkv.weight"):
                p._muon_split = 3
            elif n.endswith("mlp.fc.weight") and a.act == "swiglu":
                p._muon_split = 2
        elif p.ndim == 2 and ("wte" in n or "head" in n):
            embs.append(p)
        else:
            scal.append(p)
    opts = []
    adam_groups = [
        dict(params=embs, lr=a.lr, weight_decay=a.emb_wd),
        dict(params=scal, lr=a.lr, weight_decay=0.0),
    ]
    if a.opt == "muon":
        opts.append(Muon(mats, lr=a.muon_lr, momentum=a.muon_momentum, weight_decay=a.wd, cautious_wd=bool(a.cwd),
                         polar=bool(a.polar), normuon=bool(a.normuon)))
    else:
        adam_groups.append(dict(params=mats, lr=a.lr, weight_decay=a.wd))
    opts.append(torch.optim.AdamW(adam_groups, betas=(a.beta1, a.beta2), eps=1e-8, foreach=True))
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def main(argv=None):
    a = get_args(argv)
    torch.set_num_threads(a.threads)
    # softmax tails underflow to denormals, which are ~10x slower on x86: flush them to zero
    torch.set_flush_denormal(True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    eos = meta["eos_id"]
    train = load(a.data, "train")
    val = torch.cat([torch.tensor([eos]), load(a.data, "validation")])
    test = torch.cat([torch.tensor([eos]), load(a.data, "test")])

    cfg = GPTConfig(vocab_size=meta["vocab_size"], **{k: getattr(a, k) for k in asdict(GPTConfig()) if k != "vocab_size"})
    model = GPT(cfg)
    print(f"params: total={model.num_params(False)/1e6:.2f}M non-emb={model.num_params()/1e6:.2f}M", flush=True)
    opts = build_optimizers(model, a)
    fwd = torch.compile(model) if a.compile else model

    ema, swa, swa_n = None, None, 0
    g = torch.Generator().manual_seed(a.seed)
    if a.ctx_short:
        sampler = EpochSampler(train, a.ctx_short, a.batch * a.ctx // a.ctx_short, g)
    else:
        sampler = EpochSampler(train, a.ctx, a.batch, g)
    tok_per_step = a.batch * a.ctx
    hid_eval = torch.compile(model.hidden) if a.compile else model.hidden
    log = open(os.path.join(a.out, "log.txt"), "a")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    val_bytes_per_tok = meta["validation"]["bytes"] / meta["validation"]["tokens"]
    t0 = time.time()
    eval_time = 0.0  # periodic-eval time is excluded from the training time budget
    tr_loss, n_log, tlast = 0.0, 0, time.time()
    step, frac = 0, 0.0
    while True:
        el = time.time() - t0 - eval_time
        frac = el / a.time_budget if a.time_budget > 0 else step / a.steps
        if frac >= 1.0:
            break
        m = lr_mult(step, frac, a)
        for o in opts:
            for gr in o.param_groups:
                gr["lr"] = gr["base_lr"] * m
                if "momentum" in gr and a.mom_warmup:
                    f = min(1.0, step / a.mom_warmup)
                    gr["momentum"] = (1 - f) * 0.85 + f * a.muon_momentum
        if a.ctx_short and sampler.T != a.ctx and frac >= a.ctx_switch:
            ep = sampler.epoch
            sampler = EpochSampler(train, a.ctx, a.batch, g)
            sampler.epoch = ep
            P(f"  switching to ctx {a.ctx} at step {step}")
        x, y = sampler.next()
        loss = fwd(x, y)
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        if a.ema > 0 and frac >= a.ema_start:
            if ema is None:
                ema = [p.detach().clone() for p in model.parameters()]
            else:
                torch._foreach_lerp_(ema, [p.detach() for p in model.parameters()], 1 - a.ema)
        if a.swa_start > 0 and frac >= a.swa_start and (step + 1) % a.swa_every == 0:
            if swa is None:
                swa, swa_n = [p.detach().clone() for p in model.parameters()], 1
            else:
                swa_n += 1
                torch._foreach_lerp_(swa, [p.detach() for p in model.parameters()], 1.0 / swa_n)
        tr_loss += loss.item()
        n_log += 1
        step += 1
        if step % a.log_every == 0:
            now = time.time()
            P(f"step {step} frac {frac:.3f} ep {sampler.epoch} loss {tr_loss/n_log:.4f} lr {m:.3f} "
              f"tok/s {tok_per_step*n_log/(now-tlast):.0f} elapsed {now-t0-eval_time:.0f}s")
            tr_loss, n_log, tlast = 0.0, 0, now
        if a.eval_every and step % a.eval_every == 0:
            te0 = time.time()
            nll, n = eval_nll(fwd, val, sampler.T, sampler.T, max_tokens=a.eval_tokens, hidden_fn=hid_eval)
            bpb = nll / n / math.log(2) / val_bytes_per_tok
            P(f"  eval step {step}: val_loss {nll/n:.4f} ~val_bpb(subset) {bpb:.4f}")
            eval_time += time.time() - te0
            tlast = time.time()

    train_time = time.time() - t0 - eval_time
    stride = a.eval_stride or a.ctx // 2
    res = {"train_time": train_time, "steps": step, "epochs": step * tok_per_step / len(train),
           "params_nonemb": model.num_params(), "params_total": model.num_params(False)}
    te = time.time()
    variants = {"last": [p.detach().clone() for p in model.parameters()]}
    if ema is not None:
        variants["ema"] = ema
    if swa is not None:
        variants["swa"] = swa
    best_name, best_bpb = None, float("inf")
    for name, ws in variants.items():
        with torch.no_grad():
            for p, w in zip(model.parameters(), ws):
                p.copy_(w)
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "args": vars(a)},
                   os.path.join(a.out, f"model_{name}.pt"))
        nll, n = eval_nll(fwd, val, a.ctx, stride, hidden_fn=hid_eval)
        bpb = nll / math.log(2) / meta["validation"]["bytes"]
        res[f"val_bpb_{name}"] = bpb
        P(f"  final val bpb [{name}] {bpb:.4f}")
        if bpb < best_bpb:
            best_name, best_bpb = name, bpb
    with torch.no_grad():
        for p, w in zip(model.parameters(), variants[best_name]):
            p.copy_(w)
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "args": vars(a)}, os.path.join(a.out, "model.pt"))
    res["best"] = best_name
    res["val_bpb"] = best_bpb
    if a.final_test:
        nll, n = eval_nll(fwd, test, a.ctx, stride, hidden_fn=hid_eval)
        res["test_bpb"] = nll / math.log(2) / meta["test"]["bytes"]
        res["test_loss"] = nll / n
    res["eval_time"] = time.time() - te
    P("FINAL", json.dumps(res))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
