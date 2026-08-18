"""Bundle F -- Allen-Cahn PINN architecture study (referees R1.5, R3.4).

R1.5 notes that each benchmark uses a single architecture; R3.4 asks specifically
for width/depth scaling, Fourier features and adaptive activations. Allen-Cahn is
chosen because it is the unsteady three-dimensional case, where the PINN's
advantage over the adjoint is largest and where an architecture dependence would
therefore matter most.

Three rows of three, each row holding everything else at the production setting:

* **size**       -- small / paper / large, varying the state network U_theta and
                    the unknown network S_phi together (690 / 3458 / 17218
                    parameters, i.e. 0.2x / 1x / 5.0x).
* **activation** -- tanh (paper) / silu / adaptive, the last being layer-wise
                    L-LAAF (Jagtap, Kawaguchi and Karniadakis, JCP 2020) with a
                    tanh base, which adds one parameter per layer and is identical
                    to tanh at initialization.
* **Fourier**    -- none (paper) / F1 / F2, a random Fourier embedding on the state
                    network. F1 and F2 differ only in the embedding scale
                    (s = 1 and s = 5) at fixed embed_dim, so the two have equal
                    parameter counts and the comparison isolates the scale, which
                    is the knob Fourier features are most sensitive to.

ReLU is deliberately excluded: the Allen-Cahn residual needs d2u/dx2 and ReLU's
second derivative vanishes almost everywhere, so the PDE loss would be identically
zero rather than merely poor.

The paper configuration is shared by all three rows and is re-run here under the
same stopping rule, so the baseline is compared at convergence rather than against
bundle D's fixed-budget numbers.

Every configuration, the baseline included, is run to a plateau -- outer epochs
continue until the loss stops improving by more than 1e-4 for five consecutive
epochs, with a ceiling of 80. This matters: at the production budget of 15 epochs
the baseline is essentially flat while silu and the Fourier variants are still
improving by 10-20% in their final epochs, so a fixed budget would have ranked the
budget rather than the architecture.

Three seeds first. The Allen-Cahn PINN has a ~57% relative seed spread, so three
seeds give a standard error near 33% and resolve only differences above roughly
+-66%. Configurations that land close to the baseline are escalated to five seeds
before any claim is made about them.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cases", "case3_allencahn"))

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes, trajectory_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2)          # escalate to (0,1,2,3,4) for configurations that need it

# name -> (hidden_U, layers_U, hidden_S, layers_S, act, fourier_m, fourier_scale)
CONFIGS = {
    "paper":       (32, 3, 32, 2, "tanh",     0, 1.0),
    "size_small":  (16, 2, 16, 2, "tanh",     0, 1.0),
    "size_large":  (64, 4, 64, 2, "tanh",     0, 1.0),
    "act_silu":    (32, 3, 32, 2, "silu",     0, 1.0),
    "act_adaptive":(32, 3, 32, 2, "adaptive", 0, 1.0),
    "fourier_F1":  (32, 3, 32, 2, "tanh",     8, 1.0),
    "fourier_F2":  (32, 3, 32, 2, "tanh",     8, 5.0),
}


def rows():
    return [{"config": c, "seed": s} for c in CONFIGS for s in SEEDS]


def run_one(row, index):
    require_l40s()

    from cfg import AllenCahn3DConfig
    import PINN as ac_pinn
    import solver as ac_solver

    name = row["config"]
    hu, lu, hs, ls, act, fm, fs = CONFIGS[name]

    run_dir = os.path.join(REPO, "results", "F_runs", f"{name}_s{row['seed']}")
    os.makedirs(run_dir, exist_ok=True)

    cfg = AllenCahn3DConfig()
    cfg.hidden_U, cfg.layers_U = hu, lu
    cfg.hidden_S, cfg.layers_S = hs, ls
    cfg.act_U = cfg.act_S = act
    # The Fourier embedding is applied to the state network only; the unknown f(u)
    # is a scalar function of one variable whose target is a cubic.
    cfg.fourier_m_U, cfg.fourier_scale_U = fm, fs
    cfg.seed = row["seed"]
    # Run to a plateau, not to a fixed epoch count: at the default 15 epochs the
    # baseline is nearly flat while the slower variants are still improving by
    # 10-20% per epoch, so a fixed budget would rank the budget, not the
    # architecture.
    cfg.scipy_pinn_epochs = 80
    cfg.pinn_epoch_plateau = 5
    cfg.pinn_epoch_rtol = 1.0e-4
    cfg.pinn_scipy_path = os.path.join(run_dir, "ac3d_pinn.npz")
    cfg.pinn_scipy_model = os.path.join(run_dir, "ac3d_pinn.pt")
    set_seed(cfg.seed)

    xx, yy, zz = ac_solver.make_xyz_grid(cfg)
    u0 = ac_solver.initial_condition(xx, yy, zz)
    u_hist, t_hist, _ = ac_solver.solve_allen_cahn_imex(cfg, u0, ac_solver.reaction_true)
    uT_target = u_hist[-1]

    counters = Counters()
    counters.traj_bytes_analytic = trajectory_bytes(cfg.Nx * cfg.Ny * cfg.Nz,
                                                    len(t_hist) - 1)

    with RunTimer() as timer:
        hist = ac_pinn.train_pinn(u0, uT_target, cfg)

    U_theta, S_phi = ac_pinn.build_models(cfg)
    n_state = sum(p.numel() for p in U_theta.parameters())
    n_unknown = sum(p.numel() for p in S_phi.parameters())
    n_evals = int(len(hist["bfgs"]["loss"]))
    counters.n_params = n_unknown
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(n_unknown)
    counters.n_fev = counters.n_iter = n_evals
    counters.residual(cfg.n_interior * max(n_evals, 1))

    return make_record(
        bundle="F", benchmark="allencahn", method="pinn", representation="nn",
        optimizer="ssbroyden2",
        arch=f"{name}: U w{hu}d{lu} S w{hs}d{ls} {act}"
             + (f" fourier(m={fm},s={fs})" if fm else ""),
        seed=row["seed"], noise=0.0, gamma=0.0, beta=cfg.alpha_reg,
        eps_f=float(hist["bfgs"]["rel_l2_f"][-1]),
        eps_u=float(hist["bfgs"]["rel_l2_uT"][-1]),
        converged=(hist.get("stop_reason") == "plateau"),
        stop_reason=hist.get("stop_reason", "maxepochs"),
        **counters.as_dict(), **timer.as_dict(),
        config=name, epochs_run=hist.get("epochs_run"), n_state_params=n_state, n_unknown_params=n_unknown,
        n_total_params=n_state + n_unknown,
        act=act, fourier_m=fm, fourier_scale=fs,
        cfg_snapshot=cfg,
    )


if __name__ == "__main__":
    cli("F", rows, run_one)
