"""Chunked fused linear + cross-entropy for CPU.

Never materialises the full (N, V) logits: each row-chunk computes logits, softmax gradient,
dH and accumulates dW immediately, so the working set stays cache-resident. Gradients are
computed in the forward pass and scaled by grad_output in backward (loss is a scalar mean/sum).
"""
import torch
import torch.nn.functional as F


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, y, chunk, softcap, reduce_mean):
        N = h.size(0)
        need_grad = h.requires_grad or W.requires_grad
        dh = torch.empty_like(h) if need_grad else None
        dW = torch.zeros_like(W) if need_grad else None
        total = h.new_zeros(())
        scale = 1.0 / N if reduce_mean else 1.0
        for s in range(0, N, chunk):
            hc = h[s:s + chunk]
            yc = y[s:s + chunk]
            z = hc @ W.t()
            if softcap > 0:
                t = torch.tanh(z / softcap)
                z = softcap * t
            lse = torch.logsumexp(z, dim=-1)
            total += (lse - z.gather(1, yc[:, None]).squeeze(1)).sum()
            if need_grad:
                g = torch.exp(z - lse[:, None])
                g[torch.arange(len(yc)), yc] -= 1.0
                if softcap > 0:
                    g = g * (1 - t * t)
                g *= scale
                dh[s:s + chunk] = g @ W
                dW.addmm_(g.t(), hc)
        ctx.save_for_backward(dh, dW)
        return total * scale

    @staticmethod
    def backward(ctx, go):
        dh, dW = ctx.saved_tensors
        return dh * go, dW * go, None, None, None, None


def fused_linear_ce(h, W, y, chunk=512, softcap=0.0, reduction="mean"):
    return _FusedLinearCE.apply(h, W, y, chunk, softcap, reduction == "mean")


@torch.no_grad()
def chunked_nll(h, W, y, chunk=1024, softcap=0.0):
    """Per-token NLL without building full logits (eval)."""
    out = torch.empty(h.size(0), dtype=torch.float32)
    for s in range(0, h.size(0), chunk):
        z = h[s:s + chunk] @ W.t()
        if softcap > 0:
            z = softcap * torch.tanh(z / softcap)
        out[s:s + chunk] = torch.logsumexp(z, -1) - z.gather(1, y[s:s + chunk, None]).squeeze(1)
    return out
