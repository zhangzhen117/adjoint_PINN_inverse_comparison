"""Bundle B -- optimizer sensitivity on Burgers, the simple benchmark (referee R3.7).

R3.7 grants that SSBroyden is a reasonable choice but notes it is not yet a widely
adopted baseline, and asks whether the conclusions are expected to generalize
beyond it. Answering that needs the ranking of adjoint against PINN to be measured
under more than one optimizer.

Three stages, of which the first two are dependency-free and run together:

* **B0 -- learning-rate tuning.** {Adam, SOAP} x lr in {1e-3, 3e-3, 1e-2, 3e-2},
  one seed. This stage is not optional. On a smooth fit SOAP at lr=1e-3 is ~14x
  *worse* than Adam, while at its published default 3e-3 it is ~120x *better*: a
  single shared learning rate would measure whose default happens to suit the
  problem rather than the optimizers themselves, and a referee reading a
  single-lr comparison would be right to say so.
* **B2 -- adjoint side.** {SSBroyden2, BFGS, L-BFGS-B} x 5 seeds. These are
  line-search quasi-Newton methods with no learning rate, so they need no tuning.
  Including them is what lets R3.7 be answered for both paradigms rather than only
  for PINNs. There is no cylinder counterpart: that adjoint optimizes the single
  scalar theta = log nu, where SSBroyden's update is singular (noted in
  cylinder_config.py), so an optimizer ablation there is not meaningful.
* **B1 -- PINN side at the tuned learning rate**, 5 seeds, is a follow-up bundle
  once B0 has landed. Its SSBroyden arm is already covered by bundle A's
  pinn/nn rows at the identical configuration, so it is not re-run here.

All runs use the shared convergence rule: stop on a relative objective change below
1e-8 over 500 evaluations, or at a hard cap of twice the SSBroyden baseline
wall-clock. Runs hitting the cap are recorded converged=False and reported as
non-converged rather than silently truncated.
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
LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2)

# Measured SSBroyden baselines on this benchmark; the cap is twice these.
PINN_BASELINE_S = 190.0
ADJ_BASELINE_S = 934.0


def rows():
    out = []
    # B0: learning-rate tuning for the first-order optimizers, one seed.
    for optimizer in ("adam", "soap"):
        for lr in LR_GRID:
            out.append({"sub": "B0", "side": "pinn", "optimizer": optimizer,
                        "lr": lr, "seed": 0})
    # B2: adjoint-side optimizers, five seeds, no learning rate.
    for optimizer in ("ssbroyden2", "bfgs", "lbfgsb"):
        for seed in SEEDS:
            out.append({"sub": "B2", "side": "adjoint", "optimizer": optimizer,
                        "lr": None, "seed": seed})
    return out


def _reference(cfg):
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

    hist_dir = os.path.join(REPO, "results", "B_runs")
    os.makedirs(hist_dir, exist_ok=True)
    lr_tag = "" if row["lr"] is None else f"_lr{row['lr']:g}"
    tag = f"{row['side']}_{row['optimizer']}{lr_tag}_s{row['seed']}"

    is_pinn = row["side"] == "pinn"
    cfg = BurgersConfig(
        seed=row["seed"],
        representation="nn",
        pinn_optimizer=row["optimizer"] if is_pinn else "ssbroyden2",
        adjoint_optimizer=row["optimizer"] if not is_pinn else "ssbroyden2",
        pinn_adam_lr=row["lr"] if (is_pinn and row["lr"]) else 1e-3,
        pinn_soap_lr=row["lr"] if (is_pinn and row["lr"]) else 3e-3,
        walltime_cap_s=2.0 * (PINN_BASELINE_S if is_pinn else ADJ_BASELINE_S),
        adj_nn_path=os.path.join(hist_dir, f"adj_{tag}.npz"),
        pinn_scipy_path=os.path.join(hist_dir, f"pinn_{tag}.npz"),
        pinn_scipy_model=os.path.join(hist_dir, f"pinn_{tag}.pt"),
    )
    set_seed(cfg.seed)

    core, u0, s_true, uT_target = _reference(cfg)
    counters = Counters()
    counters.traj_bytes_analytic = trajectory_bytes(cfg.N, core.Nt)

    with RunTimer() as timer:
        if is_pinn:
            import torch
            from PINN import (MLP, load_models, make_force_net, train_pinn_dispatch)
            h = train_pinn_dispatch(u0, uT_target, cfg)
            U_theta = MLP(2, 1, cfg.hidden_U, cfg.layers_U, cfg.act_U).to(cfg.device).double()
            S_phi = make_force_net(cfg, cfg.device)
            load_models(cfg, U_theta, S_phi, cfg.pinn_scipy_model)
            with torch.no_grad():
                xt = torch.tensor(core.x, dtype=torch.float64,
                                  device=cfg.device).reshape(-1, 1)
                s_opt = S_phi(xt).cpu().numpy().ravel()
            n_params = sum(p.numel() for p in S_phi.parameters())
            counters.n_fev = h.get("nfev", 0)
            counters.n_iter = h.get("nit", 0)
            counters.residual(cfg.n_interior * max(counters.n_fev, 1))
            converged, stop_reason = h.get("converged"), h.get("stop_reason")
        else:
            import adjoint_operator as A
            phi0 = A.get_init_phi(cfg)
            phi_opt, h = A.invert_force_adjoint(u0, uT_target, cfg, phi0,
                                                n_steps_warmup=0)
            s_opt, _ = A.nn_force_forward(core.x, phi_opt, cfg)
            n_params = len(phi0)
            counters.n_fev = h.get("nfev", 0)
            counters.n_iter = h.get("nit", 0)
            counters.forward(counters.n_fev)
            counters.adjoint(counters.n_fev)
            converged, stop_reason = None, "maxiter"

    counters.n_params = n_params
    # Only the dense-Hessian methods actually allocate this; recorded for all so the
    # table can show what SSBroyden costs relative to the limited-memory fallback.
    counters.hessian_bytes_analytic = (
        ssbroyden_hessian_bytes(n_params)
        if row["optimizer"] in ("ssbroyden2", "bfgs") else 0)

    uT_rec = core.forward(u0, s_opt, store_trajectory=False)["u_final"]

    return make_record(
        bundle="B", benchmark="burgers",
        method="pinn" if is_pinn else "adjoint", representation="nn",
        optimizer=row["optimizer"],
        arch=f"w{cfg.hidden_S}d{cfg.layers_S}-{cfg.act_S}",
        seed=row["seed"], noise=0.0, gamma=0.0, beta=cfg.alpha_reg,
        eps_f=float(rel_l2_error(s_opt, s_true)),
        eps_u=float(rel_l2_error(uT_rec, uT_target)),
        converged=converged, stop_reason=stop_reason,
        **counters.as_dict(), **timer.as_dict(),
        sub=row["sub"], side=row["side"], lr=row["lr"],
        walltime_cap_s=cfg.walltime_cap_s,
        cfg_snapshot=cfg,
    )


if __name__ == "__main__":
    cli("B", rows, run_one)
