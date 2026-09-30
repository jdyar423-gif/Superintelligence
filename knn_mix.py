"""Tune the kNN semantic memory on VALIDATION and export per-token NLL arrays for mixing.

For each (k, tau) on a grid, fit the mixture weight between the model predictor (per-token NLL array,
e.g. the static stream or the dynamic-eval output) and p_knn by EM on validation; keep the best (k, tau),
then write knn NLL arrays (nats; inf where no neighbour carries the target) for validation and test.
"""
import argparse, json, math, os

import numpy as np
import torch

from knn_mem import knn_logp
from mix import em, mix_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default="data/bpe8k")
    ap.add_argument("--val_model_nll", default="", help=".npy per-token NLL of the model on validation (default: static)")
    ap.add_argument("--ks", default="8,32,128,256")
    ap.add_argument("--taus", default="0.03,0.1,0.3,1,3")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    vals = torch.load(os.path.join(a.dir, "knn_meta.pt"), weights_only=False)["vals"]
    kv = torch.load(os.path.join(a.dir, "knn_validation.pt"), weights_only=False)
    m_val = np.load(a.val_model_nll) if a.val_model_nll else kv["nll_static"].double().numpy()
    base_bpb = m_val.sum() / math.log(2) / meta["validation"]["bytes"]
    d1 = kv["D"][:, 0].median().item()
    print(f"validation model bpb {base_bpb:.4f}; median nearest dist2 {d1:.3f}")
    best = None
    for k in [int(x) for x in a.ks.split(",")]:
        for tr in [float(x) for x in a.taus.split(",")]:
            tau = tr * d1
            lp = knn_logp(kv["D"], kv["I"], vals, kv["Y"], k, tau).numpy()
            nll = np.stack([m_val, -lp])
            w = em(nll)
            bpb = mix_nll(nll, w).sum() / math.log(2) / meta["validation"]["bytes"]
            hit = np.isfinite(lp).mean()
            print(f"k={k:4d} tau={tr:5.2f}*d1 lam={w[1]:.3f} hit={hit:.3f} val_bpb={bpb:.4f}", flush=True)
            if best is None or bpb < best[0]:
                best = (bpb, k, tau, w.tolist())
    bpb, k, tau, w = best
    print("BEST", json.dumps({"val_bpb": bpb, "gain": bpb - base_bpb, "k": k, "tau": tau, "weights": w}))
    for split in ["validation", "test"]:
        f = os.path.join(a.dir, f"knn_{split}.pt")
        if os.path.exists(f):
            kd = torch.load(f, weights_only=False)
            lp = knn_logp(kd["D"], kd["I"], vals, kd["Y"], k, tau).numpy()
            np.save(os.path.join(a.dir, f"knn_nll_{split}.npy"), -lp)
            np.save(os.path.join(a.dir, f"static_nll_{split}.npy"), kd["nll_static"].double().numpy())
    json.dump({"k": k, "tau": tau, "val_bpb": bpb, "weights": w}, open(os.path.join(a.dir, "knn_best.json"), "w"))


if __name__ == "__main__":
    main()
