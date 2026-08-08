"""Bundle A -- representation x algorithm on Burgers (referee R1.4).

R1.4 asks for results where the representation is fixed and the optimization varies,
and vice versa, so the two effects can be decoupled. Burgers is the benchmark where
that is cleanest: both methods already use the identical ``MLP(1,1,32,2)`` for the
unknown, so the "nn" row holds architecture as well as representation fixed.

    representation \\ method     adjoint            PINN
    grid  (N_f = 64)             existing           new (PINN.GridForce)
    nn    (N_f = 1153)           existing           existing

Note the grid+adjoint cell starts from ``s_coarse = 0`` and has no random
initialization, so it is deterministic: it is run once and reported with n=1, while
the other three cells get five seeds. That is not a gap in the design -- it is the
sharpest form of the paper's claim that a low-dimensional grid is the fastest
deterministic choice.

The factorial controls how the *unknown* is represented, not the total parameter
count: the PINN carries its state network in every cell, so PINN+grid optimizes 64
field values alongside 1185 state-network weights. That asymmetry is intrinsic to
the paradigms and is named explicitly in the manuscript's scope section.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "Burgers_identification"))

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes, trajectory_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def rows():
    out = []
    for method in ("adjoint", "pinn"):
        for representation in ("grid", "nn"):
            # grid + adjoint has no random initialization; one run is the whole
            # distribution. Running five would report a fabricated zero variance
            # over five identical rows.
            seeds = (0,) if (method == "adjoint" and representation == "grid") else SEEDS
            for seed in seeds:
                out.append({"method": method, "representation": representation,
                            "seed": seed})
    return out


def _reference(cfg):
    """Forward-solve the truth once to build the terminal-state observation."""
    from solver import BurgersFDCore, true_force
    core = BurgersFDCore(cfg)
    u0 = np.sin(core.x)
    s_true = true_force(core.x)
    uT = core.forward(u0, s_true, store_trajectory=False)["u_final"]
    return core, u0, s_true, uT


def run_one(row, index):
    require_l40s()

    from cfg import BurgersConfig
    from solver import rel_l2_error

    hist_dir = os.path.join(REPO, "results", "A_runs")
    os.makedirs(hist_dir, exist_ok=True)
    tag = f"{row['method']}_{row['representation']}_s{row['seed']}"

    cfg = BurgersConfig(
        seed=row["seed"],
        representation=row["representation"],
        adj_nn_path=os.path.join(hist_dir, f"adj_{tag}.npz"),
        pinn_scipy_path=os.path.join(hist_dir, f"pinn_{tag}.npz"),
        pinn_scipy_model=os.path.join(hist_dir, f"pinn_{tag}.pt"),
    )
    set_seed(cfg.seed)

    core, u0, s_true, uT_target = _reference(cfg)
    counters = Counters()
    counters.traj_bytes_analytic = trajectory_bytes(cfg.N, core.Nt)

    with RunTimer() as timer:
        if row["method"] == "adjoint":
            import adjoint_operator as A
            if row["representation"] == "grid":
                _, s_opt_fine, h = A.invert_force_adjoint_grid(u0, uT_target, cfg)
                n_params = cfg.n_coarse
            else:
                phi0 = A.get_init_phi(cfg)
                phi_opt, h = A.invert_force_adjoint(u0, uT_target, cfg, phi0,
                                                    n_steps_warmup=0)
                s_opt_fine, _ = A.nn_force_forward(core.x, phi_opt, cfg)
                n_params = len(phi0)
            # One forward and one backward sweep per objective evaluation.
            counters.n_fev = h.get("nfev", 0)
            counters.n_iter = h.get("nit", 0)
            counters.forward(counters.n_fev)
            counters.adjoint(counters.n_fev)
            converged, stop_reason = None, "maxiter"
        else:
            from PINN import make_force_net, train_pinn_dispatch, MLP
            h = train_pinn_dispatch(u0, uT_target, cfg)
            # Evaluate the trained unknown on the solver grid.
            import torch
            from PINN import load_models
            U_theta = MLP(2, 1, cfg.hidden_U, cfg.layers_U, cfg.act_U).to(cfg.device).double()
            S_phi = make_force_net(cfg, cfg.device)
            load_models(cfg, U_theta, S_phi, cfg.pinn_scipy_model)
            with torch.no_grad():
                xt = torch.tensor(core.x, dtype=torch.float64,
                                  device=cfg.device).reshape(-1, 1)
                s_opt_fine = S_phi(xt).cpu().numpy().ravel()
            n_params = sum(p.numel() for p in S_phi.parameters())
            counters.n_fev = h.get("nfev", 0)
            counters.n_iter = h.get("nit", 0)
            # The PINN never runs the discrete solver during training; its work is
            # residual evaluations, which is the quantity R3.2 asks for instead.
            counters.residual(cfg.n_interior * max(counters.n_fev, 1))
            converged, stop_reason = h.get("converged"), h.get("stop_reason")

    counters.n_params = n_params
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(n_params)

    # eps_u is always measured by pushing the recovered field back through the
    # discrete solver, for both methods, so the state error is comparable.
    uT_rec = core.forward(u0, s_opt_fine, store_trajectory=False)["u_final"]
    eps_f = float(rel_l2_error(s_opt_fine, s_true))
    eps_u = float(rel_l2_error(uT_rec, uT_target))

    return make_record(
        bundle="A", benchmark="burgers",
        method=row["method"], representation=row["representation"],
        optimizer=cfg.adjoint_optimizer if row["method"] == "adjoint" else cfg.pinn_optimizer,
        arch=f"w{cfg.hidden_S}d{cfg.layers_S}-{cfg.act_S}" if row["representation"] == "nn"
             else f"grid{cfg.n_coarse}",
        seed=row["seed"], noise=0.0, gamma=0.0, beta=cfg.alpha_reg,
        eps_f=eps_f, eps_u=eps_u,
        converged=converged, stop_reason=stop_reason,
        **counters.as_dict(), **timer.as_dict(),
        cfg_snapshot=cfg,
    )


if __name__ == "__main__":
    cli("A", rows, run_one)
