"""Muon (orthogonalized momentum) + AdamW hybrid, CPU friendly."""
import torch


@torch.no_grad()
def newton_schulz5(G, steps=5, eps=1e-7):
    # quintic NS iteration from Keller Jordan's Muon; works on batched (..., m, n)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D weights. `splits` (per-param) lets fused matrices (qkv, swiglu fc) be
    orthogonalized per sub-matrix: the param of shape (k*m, n) is viewed as (k, m, n)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5,
                 cautious_wd=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps, cautious_wd=cautious_wd))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                grad = p.grad
                buf.lerp_(grad, 1 - g["momentum"])
                u = grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                k = getattr(p, "_muon_split", 1)
                if k > 1:
                    u = u.view(k, p.size(0) // k, p.size(1))
                u = newton_schulz5(u, g["ns_steps"])
                m, n = u.size(-2), u.size(-1)
                u = (u * max(1.0, m / n) ** 0.5).view_as(p)
                if g["weight_decay"] > 0:
                    if g["cautious_wd"]:
                        # decay only coordinates where the update already shrinks |p| (Cautious WD)
                        mask = (u * p) > 0
                        p.sub_(p * mask, alpha=g["lr"] * g["weight_decay"])
                    else:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u, alpha=-g["lr"])
