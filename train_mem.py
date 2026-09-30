"""Train PointerXL (streaming memory + learned pointer) on WikiText-2 on CPU; report BPB.

Training reads B contiguous streams over the train split (fresh random phase every epoch), one
segment per step, carrying detached memories between segments. Evaluation streams the split in
file order (optionally as K independent contiguous shards for speed) and scores every token exactly
once with only past context: BPB = sum NLL / ln2 / bytes.
Checkpoints are written every --ckpt_every seconds and training resumes automatically.
"""
import argparse, json, math, os, time
from dataclasses import asdict

import numpy as np
import torch

from memlm import PointerXL, MemConfig
from train import lr_mult, build_optimizers


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bpe8k")
    ap.add_argument("--out", default="runs/mem_debug")
    ap.add_argument("--seed", type=int, default=1337)
    for f, v in asdict(MemConfig()).items():
        if f == "vocab_size":
            continue
        if type(v) is bool:
            ap.add_argument(f"--{f}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{f}", type=type(v), default=v)
    ap.add_argument("--opt", default="muon")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--muon_lr", type=float, default=0.03)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--mom_warmup", type=int, default=300)
    ap.add_argument("--cwd", type=int, default=1)
    ap.add_argument("--polar", type=int, default=1)
    ap.add_argument("--normuon", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.4)
    ap.add_argument("--emb_wd", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--time_budget", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--sched", default="wsd")
    ap.add_argument("--wsd_frac", type=float, default=0.4)
    ap.add_argument("--min_lr_frac", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--ema_start", type=float, default=0.6)
    ap.add_argument("--swa_start", type=float, default=0.0)
    ap.add_argument("--swa_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_tokens", type=int, default=65536)
    ap.add_argument("--eval_shards", type=int, default=8)
    ap.add_argument("--final_shards", type=int, default=1)
    ap.add_argument("--final_test", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=float, default=900.0, help="seconds between resumable checkpoints")
    ap.add_argument("--mg_eta", type=float, default=0.0,
                    help="first-order meta-gradient: evaluate grads at theta - eta*g_prev/|g_prev| (MLP+attn mats)")
    ap.add_argument("--mg_until", type=float, default=0.9, help="progress fraction after which meta-gradient is off")
    return ap.parse_args(argv)


def load(path, split):
    return torch.from_numpy(np.load(os.path.join(path, f"{split}.npy")).astype(np.int64))


class StreamSampler:
    """B contiguous streams over the (circular) corpus; new random phase every epoch."""

    def __init__(self, data, B, T, gen):
        self.d, self.B, self.T, self.g = data, B, T, gen
        self.N = len(data)
        self.Ls = self.N // B
        self.spe = (self.Ls - 1) // T
        self.epoch, self.i, self.off = 0, self.spe, 0

    def state(self):
        return {"epoch": self.epoch, "i": self.i, "off": self.off}

    def load(self, s):
        self.epoch, self.i, self.off = s["epoch"], s["i"], s["off"]

    def next(self):
        reset = False
        if self.i >= self.spe:
            self.off = int(torch.randint(0, self.N, (1,), generator=self.g))
            self.epoch += 1
            self.i = 0
            reset = True
        starts = self.off + torch.arange(self.B) * self.Ls + self.i * self.T
        idx = (starts[:, None] + torch.arange(self.T + 1)[None]) % self.N
        w = self.d[idx]
        self.i += 1
        return w[:, :-1], w[:, 1:], reset


@torch.no_grad()
def stream_eval(fwd, model, stream, seg, shards=1, max_tokens=0, mem_len=None, ptr_len=None):
    """Score every target of `stream` (stream[0] = <eos> context) exactly once, left to right,
    with K independent contiguous shards run as a batch. Returns (sum_nll_nats, n_targets)."""
    base = getattr(model, "_orig_mod", model)
    was = base.training
    base.eval()
    n = len(stream) - 1 if not max_tokens else min(max_tokens, len(stream) - 1)
    K = shards
    Ls = -(-n // K)
    nseg = -(-Ls // seg)
    pad = torch.full((K * Ls + nseg * seg + 1,), int(stream[0]), dtype=torch.long)
    pad[: n + 1] = stream[: n + 1]
    st = base.init_state(K, mem_len, ptr_len)
    total, count = 0.0, 0
    for i in range(nseg):
        s0 = torch.arange(K) * Ls + i * seg
        idx = s0[:, None] + torch.arange(seg + 1)[None]
        w = pad[idx]
        out = fwd(w[:, :-1], w[:, 1:], st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none")
        nll = out[0]
        pos = i * seg + torch.arange(seg)[None]  # position within shard
        tgt = torch.arange(K)[:, None] * Ls + pos  # global target index
        valid = (pos < Ls) & (tgt < n)
        total += nll.double()[valid].sum().item()
        count += int(valid.sum())
        st = base.advance(st, out, seg)
    assert count == n, (count, n)
    base.train(was)
    return total, count


def main(argv=None):
    a = get_args(argv)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    eos = meta["eos_id"]
    train = load(a.data, "train")
    val = torch.cat([torch.tensor([eos]), load(a.data, "validation")])
    test = torch.cat([torch.tensor([eos]), load(a.data, "test")])
    cfg = MemConfig(vocab_size=meta["vocab_size"], **{k: getattr(a, k) for k in asdict(MemConfig()) if k != "vocab_size"})
    model = PointerXL(cfg)
    if cfg.compo_buckets:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
        model.set_ngrams([tok.id_to_token(i) for i in range(cfg.vocab_size)])
    opts = build_optimizers(model, a)
    fwd = torch.compile(model) if a.compile else model
    g = torch.Generator().manual_seed(a.seed)
    sampler = StreamSampler(train, a.batch, a.seg, g)
    state = model.init_state(a.batch)
    tok_per_step = a.batch * a.seg
    logf = open(os.path.join(a.out, "log.txt"), "a")
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    P(f"params: total={model.num_params(False)/1e6:.2f}M non-emb={model.num_params()/1e6:.2f}M "
      f"steps/epoch={sampler.spe}")
    ema, swa, swa_n, step, elapsed0 = None, None, 0, 0, 0.0
    ck_path = os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, weights_only=False)
        model.load_state_dict(ck["model"])
        for o, s in zip(opts, ck["opts"]):
            o.load_state_dict(s)
        ema, swa, swa_n, step, elapsed0 = ck["ema"], ck["swa"], ck["swa_n"], ck["step"], ck["elapsed"]
        sampler.load(ck["sampler"])
        g.set_state(ck["gen"])
        torch.set_rng_state(ck["rng"])
        state = ck["state"]
        P(f"resumed from step {step} ({elapsed0:.0f}s)")

    def save_ckpt(elapsed):
        tmp = ck_path + ".tmp"
        torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts], "ema": ema, "swa": swa,
                    "swa_n": swa_n, "step": step, "elapsed": elapsed, "sampler": sampler.state(),
                    "gen": g.get_state(), "rng": torch.get_rng_state(), "state": state}, tmp)
        os.replace(tmp, ck_path)

    mg_params = [p for n_, p in model.named_parameters() if p.ndim == 2 and "wte" not in n_ and "compo" not in n_]
    g_prev = None
    t0 = time.time() - elapsed0
    eval_time, last_ck = 0.0, time.time()
    tr_loss, n_log, tlast = 0.0, 0, time.time()
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
        x, y, reset = sampler.next()
        if reset:
            state = model.reset(state)
        use_mg = a.mg_eta > 0 and g_prev is not None and frac < a.mg_until
        if use_mg:  # look-ahead along the previous segment's (normalised) gradient
            with torch.no_grad():
                torch._foreach_add_(mg_params, g_prev, alpha=-a.mg_eta * m)
        out = fwd(x, y, state["mk"], state["mv"], state["n_mem"], state["pk"], state["py"], state["n_ptr"])
        loss = out[0]
        loss.backward()
        if use_mg:
            with torch.no_grad():
                torch._foreach_add_(mg_params, g_prev, alpha=a.mg_eta * m)
        if a.mg_eta > 0:
            with torch.no_grad():
                gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in mg_params]
                gn = torch.sqrt(sum((gg * gg).sum() for gg in gs)) + 1e-12
                g_prev = [gg / gn for gg in gs]
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        state = model.advance(state, out, a.seg)
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
            nll, n = stream_eval(fwd, model, val, a.seg, a.eval_shards, a.eval_tokens)
            bpb = nll / math.log(2) / (meta["validation"]["bytes"] * n / meta["validation"]["tokens"])
            P(f"  eval step {step}: val_loss {nll/n:.4f} ~val_bpb(subset) {bpb:.4f}")
            eval_time += time.time() - te0
            tlast = time.time()
        if time.time() - last_ck > a.ckpt_every:
            ts = time.time()
            save_ckpt(ts - t0 - eval_time)
            last_ck = time.time()
            eval_time += last_ck - ts  # checkpoint I/O does not count as training time

    train_time = time.time() - t0 - eval_time
    save_ckpt(train_time)
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
        nll, n = stream_eval(fwd, model, val, a.seg, a.final_shards)
        bpb = nll / math.log(2) / meta["validation"]["bytes"]
        res[f"val_bpb_{name}"] = bpb
        P(f"  final val bpb [{name}] {bpb:.4f}")
        if bpb < best_bpb:
            best_name, best_bpb = name, bpb
    with torch.no_grad():
        for p, w in zip(model.parameters(), variants[best_name]):
            p.copy_(w)
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "args": vars(a)}, os.path.join(a.out, "model.pt"))
    res["best"], res["val_bpb"] = best_name, best_bpb
    if a.final_test:
        nll, n = stream_eval(fwd, model, test, a.seg, a.final_shards)
        res["test_bpb"] = nll / math.log(2) / meta["test"]["bytes"]
    res["eval_time"] = time.time() - te
    P("FINAL", json.dumps(res))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
