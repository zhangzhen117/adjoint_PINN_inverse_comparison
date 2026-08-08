"""Validate the Burgers seed plumbing and the new grid representation.

Three things must hold before bundle A can be trusted:

1. ``GridForce`` reproduces the adjoint's periodic linear prolongation matrix ``P``
   exactly, so the "grid" cell of the 2x2 uses the same representation on both
   sides and the comparison is about the algorithm, not the interpolant.
2. Gradients flow to the nodal values.
3. ``set_seed`` makes both the PINN and the adjoint reproducible, and different
   seeds actually give different initializations (otherwise the seed study would
   report a spuriously tiny spread).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "Burgers_identification"))
sys.path.insert(0, REPO)

from cfg import BurgersConfig
from PINN import GridForce, make_force_net, MLP, train_pinn
from adjoint_operator import get_init_phi
from solver import BurgersFDCore, true_force, rel_l2_error
from common.seeding import set_seed


def build_interp_matrix(x_fine, L, N_f):
    """The adjoint's prolongation, copied verbatim from run_identification.ipynb."""
    N = len(x_fine)
    dx_c = L / N_f
    P = np.zeros((N, N_f))
    for i, xf in enumerate(x_fine):
        j = int(xf / dx_c) % N_f
        j1 = (j + 1) % N_f
        xl = j * dx_c
        w1 = (xf - xl) / dx_c
        w0 = 1.0 - w1
        P[i, j] += w0
        P[i, j1] += w1
    return P


cfg = BurgersConfig()
print(f"cfg: N={cfg.N} L={cfg.L:.6f} seed={cfg.seed} rep={cfg.representation} n_coarse={cfg.n_coarse}")

# ---- 1. GridForce == P @ s_coarse -------------------------------------------
x_fine = np.linspace(0, cfg.L, cfg.N, endpoint=False, dtype=np.float64)
P = build_interp_matrix(x_fine, cfg.L, cfg.n_coarse)

rng = np.random.default_rng(7)
s_coarse = rng.standard_normal(cfg.n_coarse)

gf = GridForce(cfg.n_coarse, cfg.L).double()
with torch.no_grad():
    gf.s.copy_(torch.tensor(s_coarse, dtype=torch.float64))
x_t = torch.tensor(x_fine, dtype=torch.float64).reshape(-1, 1)
got = gf(x_t).detach().numpy().ravel()
want = P @ s_coarse
print(f"\n1. GridForce vs adjoint P:  max|diff| = {np.max(np.abs(got - want)):.3e}")
assert np.allclose(got, want, atol=1e-14), "GridForce does not match the adjoint prolongation"

# off-grid points too (the PINN samples collocation points anywhere)
x_off = torch.tensor(rng.uniform(0, cfg.L, 500), dtype=torch.float64).reshape(-1, 1)
off = gf(x_off).detach().numpy().ravel()
ref = np.interp(np.asarray(x_off).ravel() % cfg.L,
                np.append(np.arange(cfg.n_coarse) * cfg.L / cfg.n_coarse, cfg.L),
                np.append(s_coarse, s_coarse[0]))
print(f"   off-grid vs np.interp:   max|diff| = {np.max(np.abs(off - ref)):.3e}")
assert np.allclose(off, ref, atol=1e-12)

# ---- 2. gradients reach the nodal values -------------------------------------
gf.zero_grad()
gf(x_t).pow(2).sum().backward()
g = gf.s.grad
print(f"\n2. grad wrt nodal values: shape {tuple(g.shape)}  |g| = {g.norm():.4e}  "
      f"all finite = {bool(torch.isfinite(g).all())}")
assert g.shape == (cfg.n_coarse,) and torch.isfinite(g).all() and g.norm() > 0

# ---- 3. parameter counts (the 2x2 axes) --------------------------------------
n_nn = sum(p.numel() for p in make_force_net(BurgersConfig(representation="nn"), "cpu").parameters())
n_gr = sum(p.numel() for p in make_force_net(BurgersConfig(representation="grid"), "cpu").parameters())
print(f"\n3. N_f: nn = {n_nn}, grid = {n_gr}")

# ---- 4. seeding ---------------------------------------------------------------
a = get_init_phi(BurgersConfig(seed=0, device="cpu"))
b = get_init_phi(BurgersConfig(seed=0, device="cpu"))
c = get_init_phi(BurgersConfig(seed=1, device="cpu"))
print(f"\n4. adjoint init: seed0==seed0 -> {np.allclose(a, b)}   "
      f"seed0==seed1 -> {np.allclose(a, c)}  (want True, False)")
assert np.allclose(a, b) and not np.allclose(a, c)

set_seed(3)
p1 = [p.detach().clone() for p in MLP(1, 1, 32, 2).parameters()]
set_seed(3)
p2 = [p.detach().clone() for p in MLP(1, 1, 32, 2).parameters()]
set_seed(4)
p3 = [p.detach().clone() for p in MLP(1, 1, 32, 2).parameters()]
same = all(torch.allclose(x, y) for x, y in zip(p1, p2))
diff = any(not torch.allclose(x, y) for x, y in zip(p1, p3))
print(f"   torch MLP:   seed3==seed3 -> {same}   seed3!=seed4 -> {diff}  (want True, True)")
assert same and diff

# ---- 5. short end-to-end PINN run in both representations --------------------
core = BurgersFDCore(cfg)
u0 = np.sin(core.x)
s_true = true_force(core.x)
uT_target = core.forward(u0, s_true, store_trajectory=False)["u_final"]
print(f"\n5. forward solve OK: ||uT|| = {np.linalg.norm(uT_target):.6f}")

for rep in ("nn", "grid"):
    small = BurgersConfig(
        representation=rep, seed=0,
        scipy_pinn_adam_warmup_steps=20, scipy_pinn_maxiter=5, scipy_pinn_epochs=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    small.adj_nn_path = f"/tmp/_smoke_adj_{rep}.npz"
    small.pinn_scipy_path = f"/tmp/_smoke_pinn_{rep}.npz"
    small.pinn_scipy_model = f"/tmp/_smoke_pinn_{rep}.pt"
    h = train_pinn(u0, uT_target, small)
    print(f"   rep={rep:4s}  runtime={h['runtime_sec']:.1f}s  "
          f"final rel_l2_f={h['bfgs']['rel_l2_f'][-1]:.4e}")

print("\nALL CHECKS PASSED")
