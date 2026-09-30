"""Muon (orthogonalized momentum) for 2-D hidden weights, fp32 and CPU friendly.

Options:
  polar    : Polar Express per-iteration coefficients (Amsel et al. 2025) instead of the fixed
             quintic (3.4445, -4.7750, 2.0315).
  normuon  : NorMuon (2025) row-wise second-moment normalisation of the orthogonalised update,
             rescaled back to the pre-normalisation Frobenius norm.
  cautious_wd : decay only coordinates where the update already shrinks |p|.
"""
import torch

POLAR = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.no_grad()
def newton_schulz5(G, steps=5, eps=1e-7, polar=False):
    # batched over leading dims (..., m, n)
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr:
        X = X.mT
    if polar:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        coeffs = POLAR[:steps] + [POLAR[-1]] * max(0, steps - len(POLAR))
    else:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
        coeffs = [(3.4445, -4.7750, 2.0315)] * steps
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D weights. A param with attribute `_muon_split=k` (fused qkv / swiglu fc) of shape
    (k*m, n) is orthogonalised as k separate (m, n) blocks."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5,
                 cautious_wd=False, polar=False, normuon=False, beta2=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                                      ns_steps=ns_steps, cautious_wd=cautious_wd, polar=polar,
                                      normuon=normuon, beta2=beta2))

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
                u = newton_schulz5(u, g["ns_steps"], polar=g["polar"])
                m, n = u.size(-2), u.size(-1)
                if g["normuon"]:
                    if "v" not in st:
                        st["v"] = torch.zeros(u.shape[:-1] + (1,))
                    v = st["v"]
                    v.lerp_(u.square().mean(dim=-1, keepdim=True), 1 - g["beta2"])
                    nrm = u.norm(dim=(-2, -1), keepdim=True)
                    u = u * v.clamp_min(1e-10).rsqrt()
                    u = u * (nrm / (u.norm(dim=(-2, -1), keepdim=True) + 1e-10))
                u = (u * max(1.0, m / n) ** 0.5).view_as(p)
                if g["weight_decay"] > 0:
                    if g["cautious_wd"]:
                        mask = (u * p) > 0
                        p.sub_(p * mask, alpha=g["lr"] * g["weight_decay"])
                    else:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u, alpha=-g["lr"])
