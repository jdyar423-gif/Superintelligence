"""Mix per-token predictive probabilities of several causal predictors (e.g. independently
dynamically-evaluated models). Mixture weights are fitted by EM on VALIDATION arrays only and then
applied unchanged to the TEST arrays. Each input .npy holds per-token NLL in nats for one predictor,
aligned on the same token stream, so the mixture is itself a valid causal distribution.

Usage: python mix.py --val a_val.npy b_val.npy --test a_test.npy b_test.npy --data data/bpe8k
"""
import argparse, json, math, os
import numpy as np


def em(nll, iters=200):
    """nll: (K, N). Returns mixture weights maximising sum log sum_k w_k exp(-nll_k)."""
    K = nll.shape[0]
    w = np.full(K, 1.0 / K)
    lp = -nll
    for _ in range(iters):
        a = np.log(w)[:, None] + lp
        m = a.max(0)
        post = np.exp(a - m)
        post /= post.sum(0)
        w = post.mean(1)
    return w


def mix_nll(nll, w):
    a = np.log(np.maximum(w, 1e-300))[:, None] - nll
    m = a.max(0)
    return -(m + np.log(np.exp(a - m).sum(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--data", default="data/bpe8k")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    V = np.stack([np.load(f) for f in a.val])
    w = em(V)
    out = {"weights": w.tolist()}
    for i, f in enumerate(a.val):
        out[f"val_bpb[{i}]"] = V[i].sum() / math.log(2) / meta["validation"]["bytes"]
    out["val_bpb_mix"] = mix_nll(V, w).sum() / math.log(2) / meta["validation"]["bytes"]
    if a.test:
        T = np.stack([np.load(f) for f in a.test])
        for i, f in enumerate(a.test):
            out[f"test_bpb[{i}]"] = T[i].sum() / math.log(2) / meta["test"]["bytes"]
        out["test_bpb_mix"] = mix_nll(T, w).sum() / math.log(2) / meta["test"]["bytes"]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
