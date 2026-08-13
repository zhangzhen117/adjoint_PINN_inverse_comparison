"""Bundle G -- does converged reference data improve the cylinder PINN's nu?

Test 4's observations come from the paper's FEM solver on its production mesh. The
OpenFOAM grid study (cylinder_gridstudy/) put the discretization error of that data
at about 0.4% on the probe vector and 1.4e-3 on the terminal field, while the
published inversion reports eps_nu = 5.2e-3 and the five-seed value is 8.8e-2. The
published number therefore sits at or below the noise floor of the data it was
fitted to, which raises an obvious question: is the PINN limited by the data, or by
its own training?

This bundle answers it by holding everything about the PINN fixed -- the paper's
32x3 tanh network, Adam -> SSBroyden, same losses, same collocation budget, same
cold start nu0 = 1.0 -- and swapping only the reference data:

    fem  : history/saturated.npz     + history/probe_obs.npz      (as published)
    of   : history/saturated_of.npz  + history/probe_obs_of.npz   (OpenFOAM L5,
                                       195432 cells, ~0.4% observation error)

Both arms are run here rather than reusing bundle D for the FEM arm, so the two go
through identical code and differ only in the input files.

The two windows are not the same physical state: the FEM window starts at its own
saturated phase and the OpenFOAM window at t = 80 of its run, and the limit cycle
has no preferred origin. So the comparison is between two equally valid instances
of the same inverse problem, not between two discretizations of one instance. That
is the right comparison for the question being asked -- whether better data buys a
better nu -- but it does mean a per-seed difference is not meaningful; only the
distributions are.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
SOURCES = {
    "fem": ("saturated.npz", "probe_obs.npz"),
    "of":  ("saturated_of.npz", "probe_obs_of.npz"),
}


def rows():
    return [{"source": s, "seed": k} for s in SOURCES for k in SEEDS]


def run_one(row, index):
    require_l40s()
    sys.path.insert(0, os.path.join(REPO, "cylinder"))
    from cylinder_config import CylinderRunConfig
    import cylinder_pinn_inverse as cpi

    src, seed = row["source"], row["seed"]
    sat, obs = SOURCES[src]

    run_dir = os.path.join(REPO, "results", "G_runs", f"cyl_{src}_s{seed}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = CylinderRunConfig()
    cfg.pinn_setup = "paper"          # the published setup, explicitly
    cfg.seed_inv = 44 + seed
    cyl_hist = os.path.join(REPO, "cylinder", "history")
    cfg.saturated_path = os.path.join(cyl_hist, sat)
    cfg.obs_path = os.path.join(cyl_hist, obs)
    cfg.hist_dir = run_dir
    cfg.fig_dir = fig_dir
    cfg.pinn_inv_npz = os.path.join(run_dir, "pinn_inv.npz")
    cfg.pinn_inv_model = os.path.join(run_dir, "pinn_inv_model.pt")

    set_seed(cfg.seed_inv)
    counters = Counters()
    with RunTimer() as timer:
        out = cpi.train(cfg)

    n_params = sum(p.numel() for p in out["model"].parameters())
    n_evals = len(out["hist"]["loss"])
    counters.n_params = n_params
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(n_params)
    counters.n_fev = counters.n_iter = n_evals
    counters.residual(cfg.n_pde * max(n_evals, 1))

    # The viscosity trajectory is kept so the loss/nu decoupling seen elsewhere on
    # this benchmark can be checked here too: the run may pass through a good nu
    # and drift away from it, in which case the final value understates what the
    # data supports.
    nu_hist = np.asarray(out["hist"].get("nu", []), float)
    nu_true = float(out["nu_true"])
    best = (float(np.abs(nu_hist / nu_true - 1.0).min())
            if nu_hist.size else float("nan"))

    return make_record(
        bundle="G", benchmark="cylinder", method="pinn", representation="scalar",
        optimizer="ssbroyden2", arch=f"w{cfg.hidden}d{cfg.layers}-tanh",
        seed=cfg.seed_inv, noise=cfg.obs_noise, gamma=0.0, beta=0.0,
        eps_f=float(out["rel_err"]), eps_u=None,
        converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        source=src, config=src,
        nu_rec=float(out["nu_rec"]), nu_true=nu_true,
        eps_nu_best_iterate=best,
        cfg_snapshot=cfg.snapshot(),
    )


if __name__ == "__main__":
    cli("G", rows, run_one)
