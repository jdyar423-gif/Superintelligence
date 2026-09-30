"""Semantic (non-parametric) memory for PointerXL: kNN retrieval over the whole TRAIN split.

Datastore: for every train position i, key = final hidden state h_i of the trained model (streamed
with its episodic memory), value = the next token x_{i+1}. Keys are PCA-projected (fit on train).
Query at test position t = h_t of the same (static) model, computed from tokens <= t only.
    p_knn(w | x_<=t) = sum_{i in kNN(h_t), v_i = w} softmax_i(-||q - k_i||^2 / tau)
This is a proper distribution built from train only, hence strictly causal. The per-token log p_knn of
the target is saved for mixing with the (dynamically evaluated) model via EM on validation (mix.py).
"""
import argparse, json, math, os, time

import numpy as np
import torch

from dyneval_mem import load_model, split_stream


@torch.no_grad()
def hidden_states(model, stream, seg, shards, mem_len=None, ptr_len=None):
    """Stream `stream` (stream[0] = context token) as K contiguous shards; return (H, Y, NLL) for every
    target in order: H (n, d) final hidden, Y (n,) targets, NLL (n,) model NLL of the target."""
    model.eval()
    n = len(stream) - 1
    K = shards
    Ls = -(-n // K)
    nseg = -(-Ls // seg)
    pad = torch.full((K * Ls + nseg * seg + 1,), int(stream[0]), dtype=torch.long)
    pad[: n + 1] = stream
    d = model.cfg.d_model
    H = torch.empty(K, nseg * seg, d)
    NL = torch.empty(K, nseg * seg)
    st = model.init_state(K, mem_len, ptr_len)
    for i in range(nseg):
        s0 = torch.arange(K) * Ls + i * seg
        w = pad[s0[:, None] + torch.arange(seg + 1)[None]]
        out = model(w[:, :-1], w[:, 1:], st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none_h")
        H[:, i * seg:(i + 1) * seg] = out[5]
        NL[:, i * seg:(i + 1) * seg] = out[0]
        st = model.advance(st, out, seg)
    H = H[:, :Ls].reshape(K * Ls, d)[:n]
    NL = NL[:, :Ls].reshape(K * Ls)[:n]
    return H, stream[1:n + 1].clone(), NL


@torch.no_grad()
def knn_search(Q, Kt, kn, q_chunk=2048, k_chunk=262144):
    """Exact kNN by squared L2. Q (nq, r), Kt (N, r). Returns (dist2 (nq, kn), idx (nq, kn))."""
    kk = (Kt * Kt).sum(1)
    D_all, I_all = [], []
    for qs in range(0, Q.size(0), q_chunk):
        q = Q[qs:qs + q_chunk]
        qq = (q * q).sum(1, keepdim=True)
        bestD = torch.full((q.size(0), kn), float("inf"))
        bestI = torch.zeros((q.size(0), kn), dtype=torch.long)
        for ks in range(0, Kt.size(0), k_chunk):
            kc = Kt[ks:ks + k_chunk]
            d2 = qq - 2.0 * (q @ kc.t()) + kk[ks:ks + k_chunk][None]
            dv, di = torch.topk(d2, min(kn, kc.size(0)), dim=1, largest=False)
            D = torch.cat([bestD, dv], 1)
            I = torch.cat([bestI, di + ks], 1)
            bestD, sel = torch.topk(D, kn, dim=1, largest=False)
            bestI = I.gather(1, sel)
        D_all.append(bestD)
        I_all.append(bestI)
    return torch.cat(D_all), torch.cat(I_all)


def knn_logp(D, I, vals, y, k, tau):
    """log p_knn(y) using the k nearest (of the stored neighbours) with temperature tau."""
    d = D[:, :k].double()
    lw = torch.log_softmax(-d / tau, dim=1)
    hit = vals[I[:, :k]] == y[:, None]
    lw = lw.masked_fill(~hit, float("-inf"))
    return torch.logsumexp(lw, dim=1)  # -inf where no neighbour carries the target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output directory for datastore / per-token arrays")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--kn", type=int, default=256, help="neighbours stored per query")
    ap.add_argument("--train_shards", type=int, default=64)
    ap.add_argument("--query_shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0, help="debug: cap datastore size")
    ap.add_argument("--max_query", type=int, default=0, help="debug: cap query tokens")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    os.makedirs(a.out, exist_ok=True)
    model, ck = load_model(a.ckpt)
    cfg = ck["cfg"]
    data = ck["args"]["data"]
    meta = json.load(open(os.path.join(data, "meta.json")))
    seg = cfg["seg"]
    t0 = time.time()
    train = torch.from_numpy(np.load(os.path.join(data, "train.npy")).astype(np.int64))
    if a.max_train:
        train = train[: a.max_train]
    train_s = torch.cat([torch.tensor([meta["eos_id"]]), train])
    Htr, Ytr, _ = hidden_states(model, train_s, seg, a.train_shards)
    print(f"datastore: {Htr.shape} in {time.time()-t0:.0f}s", flush=True)
    mu = Htr.mean(0)
    C = torch.cov((Htr - mu).t())
    evals, evecs = torch.linalg.eigh(C)
    Pm = evecs[:, -a.pca:] if a.pca < Htr.size(1) else torch.eye(Htr.size(1))
    Ktr = ((Htr - mu) @ Pm).contiguous()
    del Htr
    for split in a.splits.split(","):
        t1 = time.time()
        s = split_stream(data, split, meta["eos_id"])
        if a.max_query:
            s = s[: a.max_query + 1]
        Hq, Yq, NLq = hidden_states(model, s, seg, a.query_shards)
        Q = ((Hq - mu) @ Pm).contiguous()
        D, I = knn_search(Q, Ktr, a.kn)
        torch.save({"D": D, "I": I, "Y": Yq, "nll_static": NLq}, os.path.join(a.out, f"knn_{split}.pt"))
        print(f"{split}: static bpb {NLq.double().sum().item()/math.log(2)/meta[split]['bytes']:.4f} "
              f"search {time.time()-t1:.0f}s", flush=True)
    torch.save({"vals": Ytr, "mu": mu, "P": Pm}, os.path.join(a.out, "knn_meta.pt"))


if __name__ == "__main__":
    main()
