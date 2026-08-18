"""Bundle H -- Allen-Cahn PINN-warm-started adjoint restart, five seeds.

Section 3.3 reports the hybrid restart from a single run: warm started from one
PINN estimate at eps_f = 2.38e-3, the adjoint reached 2.88e-4 in 2448 s, below the
cold-started baseline. Two things make that worth repeating across seeds.

First, the cold baseline is now a distribution, 8.30 +- 6.6e-4 over five seeds, so
a single restart at 2.88e-4 sits within one standard deviation of it: on the
evidence of one run, "lands the adjoint in a better minimum than a cold start"
cannot be distinguished from seed luck.

Second, the same experiment on the cylinder came out the other way. There all five
restarts converged to exactly the cold-start viscosity, so the warm start bought
cost and not accuracy. Whether Allen-Cahn behaves the same way is a question about
whether the high-dimensional weight-space problem has one basin or several, which
is precisely the claim Section 3.3 makes about the adjoint occasionally reaching a
distinctly worse minimum.

Each restart is warm started from the corresponding bundle-D PINN, i.e. the same
runs Table 2 reports for the Allen-Cahn PINN (15 epochs x 200 quasi-Newton
iterations = 3000, eps_f = 3.88 +- 2.2e-3). The plateau-run variant of Appendix B.2
reaches 1.27e-3 and would make a better warm start, but it is not the arm the
table reports, and the restart should follow the PINN the paper presents.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes, trajectory_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def rows():
    return [{"seed": s} for s in SEEDS]


def run_one(row, index):
    require_l40s()
    sys.path.insert(0, os.path.join(REPO, "cases", "case3_allencahn"))
    from cfg import AllenCahn3DConfig
    import adjoint_operator as A
    import solver as ac_solver
    import PINN as ac_pinn

    seed = row["seed"]
    run_dir = os.path.join(REPO, "results", "H_runs", f"ac_restart_s{seed}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = AllenCahn3DConfig()
    cfg.seed = seed
    cfg.adj_nn_path = os.path.join(run_dir, "ac3d_adj_restart.npz")
    cfg.adj_nn_model = os.path.join(run_dir, "ac3d_adj_restart.pt")
    set_seed(cfg.seed)

    xx, yy, zz = ac_solver.make_xyz_grid(cfg)
    u0 = ac_solver.initial_condition(xx, yy, zz)
    u_hist, t_hist, _ = ac_solver.solve_allen_cahn_imex(cfg, u0,
                                                        ac_solver.reaction_true)
    uT_target = u_hist[-1]
    n_steps = len(t_hist) - 1

    # ---- warm start: this seed's PINN reaction network ----
    pinn_pt = os.path.join(REPO, "results", "D_runs", f"ac_s{seed}", "ac3d_pinn.pt")
    if not os.path.exists(pinn_pt):
        raise FileNotFoundError(f"no PINN warm start for seed {seed}: {pinn_pt}")
    _, S_phi = ac_pinn.build_models(cfg)
    # PINN.py saves {"U_state_dict", "S_state_dict", "cfg"}; the adjoint optimizes
    # the reaction network only, so take S.
    state = torch.load(pinn_pt, map_location=cfg.device, weights_only=False)
    S_phi.load_state_dict(state["S_state_dict"])
    phi0 = np.concatenate([p.detach().cpu().numpy().ravel()
                           for p in S_phi.parameters()]).astype(float)

    counters = Counters()
    counters.traj_bytes_analytic = trajectory_bytes(cfg.Nx * cfg.Ny * cfg.Nz, n_steps)

    # n_steps_warmup=0, unlike the cold start. The Adam warmup exists to pull a
    # random initialization into a sane region; applied to a warm start it does the
    # opposite. Measured on the first attempt at this bundle, 20 Adam steps at
    # lr=1e-3 took seed 0 from the PINN's eps_f = 2.0e-3 to 4.8e-1 after one step
    # and 25.3 by step 21, so the quasi-Newton phase then re-converged from a point
    # far worse than where it was handed. The published restart went straight into
    # the quasi-Newton phase, and this reproduces that.
    with RunTimer() as timer:
        phi_opt, hist = A.invert_force_adjoint(u0, uT_target, cfg, phi0,
                                               n_steps_warmup=0)

    ev = hist.get("evals", [])
    it = hist.get("iters", [])
    eps_f = float(ev[-1]["rel_l2_f"]) if ev else float("nan")
    eps_u = float(ev[-1]["rel_l2_uT"]) if ev else float("nan")
    eps_f0 = float(ev[0]["rel_l2_f"]) if ev else float("nan")   # the warm start

    counters.n_params = len(phi0)
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(len(phi0))
    counters.n_fev = len(ev)
    counters.n_iter = len(it)
    counters.forward(len(ev))
    counters.adjoint(len(ev))

    return make_record(
        bundle="H", benchmark="allencahn", method="adjoint_restart",
        representation="nn", optimizer=cfg.scipy_method_bfgs,
        arch=f"w{cfg.hidden_S}d{cfg.layers_S}-{cfg.act_S}",
        seed=seed, noise=0.0, gamma=0.0, beta=cfg.alpha_reg,
        eps_f=eps_f, eps_u=eps_u, converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        eps_f_warmstart=eps_f0, config="restart",
        cfg_snapshot=cfg,
    )


if __name__ == "__main__":
    cli("H", rows, run_one)
