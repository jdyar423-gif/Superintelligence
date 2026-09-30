"""Correctness tests for PointerXL: causality and normalisation of the pointer mixture."""
import torch
from memlm import PointerXL, MemConfig

torch.manual_seed(0)
V = 40
cfg = MemConfig(vocab_size=V, n_layer=2, n_head=2, d_model=32, seg=8, mem_len=8, ptr_len=24, ptr_dim=8,
                max_pos=64, softcap=15.0)
m = PointerXL(cfg).eval()
with torch.no_grad():  # make every path active (zero-init projections would hide leaks)
    for p in m.parameters():
        p.add_(0.3 * torch.randn_like(p))


def run_stream(tokens, seg=8):
    """Score tokens[1:] given tokens[:-1] with streaming memory; returns per-target nll."""
    st = m.init_state(1)
    out_nll = []
    n = len(tokens) - 1
    for s in range(0, n, seg):
        x = tokens[s:s + seg][None]
        y = tokens[s + 1:s + seg + 1][None]
        L = x.size(1)
        if L < seg:
            break
        out = m(x, y, st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none")
        out_nll.append(out[0][0])
        st = m.advance(st, out, seg)
    return torch.cat(out_nll)


with torch.no_grad():
    # 1) causality: changing token k must not change nll of targets < k-1 (target t is token t+1)
    toks = torch.randint(0, V, (8 * 6 + 1,))
    base = run_stream(toks)
    worst = 0.0
    for k in range(1, len(toks)):
        t2 = toks.clone()
        t2[k] = (t2[k] + 7) % V
        alt = run_stream(t2)
        # targets 0..k-2 only depend on tokens <= k-1
        if k - 1 > 0:
            worst = max(worst, (alt[:k - 1] - base[:k - 1]).abs().max().item())
    print("causality max |delta| on past targets:", worst)
    assert worst < 1e-5

    # 2) normalisation: sum_w p(w) = 1 at every position of a segment deep in the stream
    toks = torch.randint(0, 6, (8 * 4 + 1,))  # small alphabet -> many pointer matches
    st = m.init_state(1)
    for s in range(0, 24, 8):
        x, y = toks[s:s + 8][None], toks[s + 1:s + 9][None]
        out = m(x, y, st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none")
        st = m.advance(st, out, 8)
    x = toks[24:32][None]
    for t in range(8):
        tot = 0.0
        for w in range(V):
            y = toks[25:33].clone()[None]
            y[0, t] = w
            # later targets may change, but position t only sees targets < t as pointer values
            nll = m(x, y, st["mk"], st["mv"], st["n_mem"], st["pk"], st["py"], st["n_ptr"], "none")[0][0, t]
            tot += torch.exp(-nll).item()
        assert abs(tot - 1) < 1e-4, (t, tot)
    print("normalisation OK (sum p = 1 at all positions)")
