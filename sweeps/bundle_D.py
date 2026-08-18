"""Bundle D -- multi-initialization statistics (referees R1.7, R3.5).

R1.7 asks for multi-initialization statistics for all PINN experiments and R3.5 for
mean, standard deviation and results across multiple random seeds, since PINNs are
known to exhibit training variability. Five seeds per configuration, everywhere.

This module covers the two benchmarks not already seeded by another bundle:
the cylinder PINN and the Allen-Cahn PINN. Burgers comes from bundle A and Darcy
from bundle E, so re-running them here would only duplicate records.

Deliberately *not* run, and stated in the manuscript instead:

* the **Darcy KL adjoint** starts from xi_0 = 0 and the **cylinder adjoint** from a
  fixed nu_0, with no random initialization anywhere -- both are deterministic, so
  a five-seed study would report a fabricated zero variance;
* the **Allen-Cahn adjoint** does have a random MLP initialization, but costs
  5469 s per run, so it stays single-seed and is flagged as such rather than
  quietly presented as if it had error bars.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes, trajectory_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def rows():
    return ([{"benchmark": "cylinder", "seed": s} for s in SEEDS]
            + [{"benchmark": "allencahn", "seed": s} for s in SEEDS])


def _run_cylinder(seed):
    sys.path.insert(0, os.path.join(REPO, "cases", "case4_cylinder"))
    from cylinder_config import CylinderRunConfig
    import cylinder_pinn_inverse as cpi

    run_dir = os.path.join(REPO, "results", "D_runs", f"cyl_s{seed}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = CylinderRunConfig()
    cfg.seed_inv = 44 + seed
    # Inputs by absolute path: the cylinder's defaults are relative to the CWD,
    # which for a sweep is the repo root. The cached 807 s warmup is reused.
    cyl_hist = os.path.join(REPO, "cases", "case4_cylinder", "history")
    cfg.saturated_path = os.path.join(cyl_hist, "saturated.npz")
    cfg.obs_path = os.path.join(cyl_hist, "probe_obs.npz")
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

    rec = make_record(
        bundle="D", benchmark="cylinder", method="pinn", representation="scalar",
        optimizer="ssbroyden2", arch=f"w{cfg.hidden}d{cfg.layers}-tanh",
        seed=cfg.seed_inv, noise=cfg.obs_noise, gamma=0.0, beta=0.0,
        eps_f=float(out["rel_err"]), eps_u=None,
        converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        nu_rec=float(out["nu_rec"]), nu_true=float(out["nu_true"]),
        cfg_snapshot=cfg.snapshot(),
    )
    return rec


def _run_allencahn(seed):
    sys.path.insert(0, os.path.join(REPO, "cases", "case3_allencahn"))
    from cfg import AllenCahn3DConfig
    import PINN as ac_pinn
    import solver as ac_solver

    run_dir = os.path.join(REPO, "results", "D_runs", f"ac_s{seed}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = AllenCahn3DConfig()
    cfg.seed = seed
    cfg.pinn_scipy_path = os.path.join(run_dir, "ac3d_pinn.npz")
    cfg.pinn_scipy_model = os.path.join(run_dir, "ac3d_pinn.pt")
    set_seed(cfg.seed)

    # Terminal-state observation from the reference reaction, same solver.
    xx, yy, zz = ac_solver.make_xyz_grid(cfg)
    u0 = ac_solver.initial_condition(xx, yy, zz)
    u_hist, t_hist, _ = ac_solver.solve_allen_cahn_imex(cfg, u0,
                                                        ac_solver.reaction_true)
    uT_target = u_hist[-1]
    n_steps = len(t_hist) - 1

    counters = Counters()
    counters.traj_bytes_analytic = trajectory_bytes(cfg.Nx * cfg.Ny * cfg.Nz, n_steps)

    with RunTimer() as timer:
        hist = ac_pinn.train_pinn(u0, uT_target, cfg)

    eps_f = float(hist["bfgs"]["rel_l2_f"][-1])
    eps_u = float(hist["bfgs"]["rel_l2_uT"][-1])
    # Parameter count of the unknown's network, matching how bundle A reports N_f.
    _, S_phi = ac_pinn.build_models(cfg)
    n_params = sum(p.numel() for p in S_phi.parameters())
    counters.n_params = n_params
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(n_params)
    n_evals = int(len(hist["bfgs"]["loss"]))
    counters.n_fev = n_evals
    counters.n_iter = n_evals
    counters.residual(cfg.n_interior * max(n_evals, 1))

    return make_record(
        bundle="D", benchmark="allencahn", method="pinn", representation="nn",
        optimizer="ssbroyden2", arch=f"w{cfg.hidden_S}d{cfg.layers_S}-{cfg.act_S}",
        seed=seed, noise=0.0, gamma=0.0, beta=cfg.alpha_reg,
        eps_f=eps_f, eps_u=eps_u, converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        cfg_snapshot=cfg,
    )


def run_one(row, index):
    require_l40s()
    if row["benchmark"] == "cylinder":
        return _run_cylinder(row["seed"])
    return _run_allencahn(row["seed"])


if __name__ == "__main__":
    cli("D", rows, run_one)
