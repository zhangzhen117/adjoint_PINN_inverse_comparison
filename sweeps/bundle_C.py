"""Bundle C -- PINN architecture sensitivity on the cylinder (referees R1.5, R3.4).

R1.5 observes that each benchmark uses a single architecture, and R3.4 asks for
architecture sensitivity naming width/depth scaling, Fourier features and adaptive
activations. The cylinder carries this study because its inverse PINN is cheap
(~190 s) and because it is the hardest of the four benchmarks.

Two sub-sweeps, deliberately small -- one small and one large setting in each
dimension rather than a dense grid:

* **C1** width in {16, 64} x depth in {2, 4}, plus the production 32x3 reference.
* **C2** encoding in {tanh (baseline), Fourier features}.

No seeds here: this measures the spread *across architectures*, and the
seed-to-seed spread at fixed architecture is what bundle D measures. Reporting
both separately keeps the two sources of variation distinguishable.

Scope note for the write-up: on this benchmark the unknown is the scalar nu, so
architecture affects the *state* network only. The matched-architecture question
(same representation of the unknown on both sides) is answered by bundle A, where
the adjoint and the PINN share an identical MLP.

The upper end of the width/depth range is bounded by SSBroyden itself: its dense
N_f x N_f inverse Hessian costs 8*N_f^2 bytes, which is the same constraint the
optimizer discussion (R3.7) is about.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cylinder"))

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

BASE_SEED = 44          # cfg.seed_inv default, kept so C is comparable with the paper


def rows():
    out = []
    # C1: one small and one large in each dimension, plus the production reference.
    for hidden, layers in ((16, 2), (16, 4), (64, 2), (64, 4), (32, 3)):
        out.append({"sub": "C1", "hidden": hidden, "layers": layers,
                    "encoding": "none"})
    # C2: baseline activation vs a Fourier feature encoding, at the production size.
    out.append({"sub": "C2", "hidden": 32, "layers": 3, "encoding": "fourier"})
    return out


def run_one(row, index):
    require_l40s()

    from cylinder_config import CylinderRunConfig
    import cylinder_pinn_inverse as cpi

    tag = f"h{row['hidden']}L{row['layers']}_{row['encoding']}"
    run_dir = os.path.join(REPO, "results", "C_runs", tag)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = CylinderRunConfig()
    cfg.hidden = row["hidden"]
    cfg.layers = row["layers"]
    cfg.pinn_encoding = row["encoding"]
    cfg.seed_inv = BASE_SEED
    # Reuse the cached warmup / observation files; only the training output moves.
    cfg.hist_dir = run_dir
    cfg.fig_dir = fig_dir
    cfg.pinn_inv_npz = os.path.join(run_dir, "pinn_inv.npz")
    cfg.pinn_inv_model = os.path.join(run_dir, "pinn_inv_model.pt")

    set_seed(cfg.seed_inv)
    counters = Counters()

    with RunTimer() as timer:
        out = cpi.train(cfg)

    model = out["model"]
    n_params = sum(p.numel() for p in model.parameters())
    counters.n_params = n_params
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(n_params)
    n_evals = len(out["hist"]["loss"])
    counters.n_fev = n_evals
    counters.n_iter = n_evals
    counters.residual(cfg.n_pde * max(n_evals, 1))

    return make_record(
        bundle="C", benchmark="cylinder",
        method="pinn", representation="scalar",
        optimizer="ssbroyden2",
        arch=f"w{row['hidden']}d{row['layers']}-{row['encoding']}",
        seed=cfg.seed_inv, noise=cfg.obs_noise, gamma=0.0, beta=0.0,
        # The unknown is the scalar nu, so eps_f is its relative error.
        eps_f=float(out["rel_err"]), eps_u=None,
        converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        sub=row["sub"], nu_rec=float(out["nu_rec"]), nu_true=float(out["nu_true"]),
        hidden=row["hidden"], layers=row["layers"], encoding=row["encoding"],
        cfg_snapshot=cfg.snapshot(),
    )


if __name__ == "__main__":
    cli("C", rows, run_one)
