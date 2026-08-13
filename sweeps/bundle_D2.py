"""Bundle D2 -- seed statistics for the Allen-Cahn adjoint (referees R1.7, R3.5).

This is the last stochastic case without seed replicates. The Allen-Cahn adjoint
optimizes an MLP whose initialization is random, so unlike the other adjoint runs it
genuinely varies from seed to seed and cannot be argued away.

Deliberately excluded, because they have no random initialization at all and a seed
study would report a fabricated zero variance rather than a measurement:

* **Burgers adjoint on the coarse grid** -- starts from s_coarse = 0.
* **Darcy KL adjoint** -- starts from xi_0 = 0.
* **Cylinder adjoint** -- a single scalar theta = log nu from a fixed nu_0.

At 5469 s per run this is the most expensive seed set in the study, which is why it
was left until the cheaper cases were settled.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "AllenCahn_3D_identification"))

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes, trajectory_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def rows():
    return [{"benchmark": "allencahn", "method": "adjoint", "seed": s} for s in SEEDS]


def run_one(row, index):
    require_l40s()

    from cfg import AllenCahn3DConfig
    import adjoint_operator as A
    import solver as ac_solver
    import PINN as ac_pinn

    run_dir = os.path.join(REPO, "results", "D_runs", f"ac_adj_s{row['seed']}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = AllenCahn3DConfig()
    cfg.seed = row["seed"]
    cfg.adj_nn_path = os.path.join(run_dir, "ac3d_adj.npz")
    cfg.adj_nn_model = os.path.join(run_dir, "ac3d_adj.pt")
    set_seed(cfg.seed)

    xx, yy, zz = ac_solver.make_xyz_grid(cfg)
    u0 = ac_solver.initial_condition(xx, yy, zz)
    u_hist, t_hist, _ = ac_solver.solve_allen_cahn_imex(cfg, u0, ac_solver.reaction_true)
    uT_target = u_hist[-1]
    n_steps = len(t_hist) - 1

    counters = Counters()
    counters.traj_bytes_analytic = trajectory_bytes(cfg.Nx * cfg.Ny * cfg.Nz, n_steps)

    _, S_phi = ac_pinn.build_models(cfg)
    phi0 = np.concatenate([p.detach().cpu().numpy().ravel()
                           for p in S_phi.parameters()]).astype(float)

    with RunTimer() as timer:
        phi_opt, hist = A.invert_force_adjoint(u0, uT_target, cfg, phi0,
                                               n_steps_warmup=20)

    ev = hist.get("evals", [])
    it = hist.get("iters", [])
    eps_f = float(ev[-1]["rel_l2_f"]) if ev and "rel_l2_f" in ev[-1] else float("nan")
    eps_u = float(ev[-1]["rel_l2_uT"]) if ev and "rel_l2_uT" in ev[-1] else float("nan")

    counters.n_params = len(phi0)
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(len(phi0))
    counters.n_fev = len(ev)
    counters.n_iter = len(it)
    counters.forward(len(ev))
    counters.adjoint(len(ev))

    return make_record(
        bundle="D", benchmark="allencahn", method="adjoint", representation="nn",
        optimizer=cfg.scipy_method_bfgs,
        arch=f"w{cfg.hidden_S}d{cfg.layers_S}-{cfg.act_S}",
        seed=row["seed"], noise=0.0, gamma=0.0, beta=cfg.alpha_reg,
        eps_f=eps_f, eps_u=eps_u, converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        cfg_snapshot=cfg,
    )


if __name__ == "__main__":
    cli("D2", rows, run_one)
