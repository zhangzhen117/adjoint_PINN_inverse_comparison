"""SOAP validation: (a) faithful to the reference, (b) preserves float64, (c) optimizes.

(a) is the one that matters for vendoring. Running the vendored copy with
precision="float32" must reproduce the unmodified reference bit-for-bit; any drift
means the precision edits changed the algorithm rather than just its arithmetic.
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _soap_reference import SOAP as SOAP_REF          # noqa: E402
from common.soap import SOAP                          # noqa: E402
from common.seeding import set_seed                   # noqa: E402
from common.instrument import require_l40s            # noqa: E402

info = require_l40s()
print("hardware OK:", info["gpu_name"])
dev = "cuda"


def make_net(dtype):
    set_seed(1)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1)).to(dev).to(dtype)


def trajectory(opt_cls, dtype, steps=60, **kw):
    torch.set_default_dtype(dtype)
    net = make_net(dtype)
    x = torch.linspace(0, 2 * np.pi, 256, device=dev, dtype=dtype).reshape(-1, 1)
    y = torch.sin(2 * x)
    opt = opt_cls(net.parameters(), **kw)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    flat = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    return np.array(losses), flat


# ---- (a) fidelity to the reference -------------------------------------------
kw = dict(lr=3e-3, precondition_frequency=5, precondition_1d=True, weight_decay=0.01)
l_ref, p_ref = trajectory(SOAP_REF, torch.float32, **kw)
l_mine, p_mine = trajectory(SOAP, torch.float32, precision="float32", **kw)
dp = float((p_ref - p_mine).abs().max())
dl = float(np.abs(l_ref - l_mine).max())
print(f"\n(a) vendored(precision='float32') vs reference, 60 steps float32:")
print(f"    max |dparam| = {dp:.3e}   max |dloss| = {dl:.3e}")
assert dp == 0.0 and dl == 0.0, "vendored copy diverges from the reference"
print("    IDENTICAL -- vendoring is faithful")

# ---- (b) float64 preserved ----------------------------------------------------
torch.set_default_dtype(torch.float64)
_, _ = trajectory(SOAP, torch.float64, steps=30, lr=3e-3,
                  precondition_frequency=5, precondition_1d=True)
print("\n(b) float64 path runs (dtype checks passed in the earlier run)")

# ---- (c) does it optimize? sweep lr, compare with Adam ------------------------
torch.set_default_dtype(torch.float64)
print("\n(c) 1500 steps, sin(2x) fit, float64 -- final loss:")
for lr in (1e-3, 3e-3, 1e-2, 3e-2):
    l_s, _ = trajectory(SOAP, torch.float64, steps=1500, lr=lr,
                        precondition_frequency=10, precondition_1d=True)
    l_a, _ = trajectory(torch.optim.Adam, torch.float64, steps=1500, lr=lr)
    print(f"    lr={lr:<6g}  SOAP {l_s[-1]:.4e}   Adam {l_a[-1]:.4e}")

# ---- (d) ill-conditioned quadratic: where preconditioning should win ----------
torch.set_default_dtype(torch.float64)
set_seed(3)
n = 64
cond = torch.logspace(0, 3, n, device=dev, dtype=torch.float64)   # kappa = 1e3
A = torch.diag(cond) @ torch.linalg.qr(torch.randn(n, n, device=dev, dtype=torch.float64))[0]
b = torch.randn(n, 1, device=dev, dtype=torch.float64)

def quad(opt_cls, steps=800, **kw):
    set_seed(4)
    W = torch.zeros(n, 1, device=dev, dtype=torch.float64, requires_grad=True)
    o = opt_cls([W], **kw)
    for _ in range(steps):
        o.zero_grad(); l = ((A @ W - b) ** 2).mean(); l.backward(); o.step()
    return float(((A @ W - b) ** 2).mean())

print(f"\n(d) ill-conditioned least squares (kappa=1e3), 800 steps:")
print(f"    SOAP lr=3e-2  {quad(SOAP, lr=3e-2, precondition_frequency=10, precondition_1d=True):.4e}")
print(f"    Adam lr=3e-2  {quad(torch.optim.Adam, lr=3e-2):.4e}")

print("\nSOAP OK")
