"""
PINN for Burgers force identification (direct residual only, no KKT).

PDE:  u_t + u*u_x - nu*u_xx = S(x),   periodic BCs.
Learn U_theta(x,t) and S_phi(x) jointly.
"""

import os
import time
from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.optimize import minimize
from numpy.linalg import cholesky, LinAlgError

from cfg import BurgersConfig
from solver import rel_l2_error, true_force

torch.set_default_dtype(torch.float64)


# ================================================================
# MLP
# ================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int, act: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        A = acts.get(act.lower(), nn.Tanh)
        net = []
        dims = [in_dim] + [hidden] * layers + [out_dim]
        for i in range(len(dims) - 2):
            net += [nn.Linear(dims[i], dims[i + 1]), A()]
        net += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*net)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ================================================================
# Batch construction
# ================================================================

def _make_fixed_batch(cfg, device, x_grid_t, u0_grid_t, x_T_t, uT_t, seed=12345):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    L, T = cfg.L, cfg.t_final

    x_int = torch.rand((cfg.n_interior, 1), generator=g, device=device, dtype=torch.float64) * L
    t_int = torch.rand((cfg.n_interior, 1), generator=g, device=device, dtype=torch.float64) * T
    t_bc = torch.rand((cfg.n_bc_t, 1), generator=g, device=device, dtype=torch.float64) * T

    return {
        "x_int": x_int, "t_int": t_int,
        "t_bc": t_bc,
        "x_ic": x_grid_t, "t_ic": torch.zeros_like(x_grid_t), "u_ic": u0_grid_t,
        "x_T": x_T_t, "t_T": torch.full_like(x_T_t, T), "u_T": uT_t,
    }


# ================================================================
# Loss functions
# ================================================================

def _pinn_loss(cfg, U_theta, S_phi, batch):
    """Direct PINN loss: PDE residual + periodic BC + IC + terminal data."""
    # PDE residual: u_t + u*u_x - nu*u_xx - S(x) = 0
    x_int = batch["x_int"].clone().requires_grad_(True)
    t_int = batch["t_int"].clone().requires_grad_(True)
    xt = torch.cat([x_int, t_int], dim=1)
    u = U_theta(xt)

    u_x = torch.autograd.grad(u, x_int, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_int, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t_int, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    Sx = S_phi(x_int)
    r = u_t + u * u_x - cfg.nu * u_xx - Sx
    L_res = torch.mean(r**2)

    # Periodic BC: u(0,t) = u(L,t) and u_x(0,t) = u_x(L,t)
    t_bc = batch["t_bc"]
    x0 = torch.zeros_like(t_bc)
    xL = torch.full_like(t_bc, cfg.L)

    u0_bc = U_theta(torch.cat([x0, t_bc], dim=1))
    uL_bc = U_theta(torch.cat([xL, t_bc], dim=1))

    x0g = x0.clone().detach().requires_grad_(True)
    xLg = xL.clone().detach().requires_grad_(True)
    tbg = t_bc.clone().detach().requires_grad_(True)
    u0g = U_theta(torch.cat([x0g, tbg], dim=1))
    uLg = U_theta(torch.cat([xLg, tbg], dim=1))
    ux0 = torch.autograd.grad(u0g, x0g, torch.ones_like(u0g), create_graph=True, retain_graph=True)[0]
    uxL = torch.autograd.grad(uLg, xLg, torch.ones_like(uLg), create_graph=True, retain_graph=True)[0]

    L_bc = torch.mean((u0_bc - uL_bc)**2) + torch.mean((ux0 - uxL)**2)

    # IC
    u_ic_pred = U_theta(torch.cat([batch["x_ic"], batch["t_ic"]], dim=1))
    L_ic = torch.mean((u_ic_pred - batch["u_ic"])**2)

    # Terminal data
    u_T_pred = U_theta(torch.cat([batch["x_T"], batch["t_T"]], dim=1))
    L_data = torch.mean((u_T_pred - batch["u_T"])**2)

    L_total = cfg.w_res * L_res + cfg.w_bc * L_bc + cfg.w_ic * L_ic + cfg.w_dataT * L_data

    Ldict = {
        "L_res": float(L_res.detach()), "L_bc": float(L_bc.detach()),
        "L_ic": float(L_ic.detach()), "L_data": float(L_data.detach()),
    }
    return L_total, Ldict


# ================================================================
# Helpers
# ================================================================

def _pack_params(models):
    params = []
    for m in models:
        params += list(m.parameters())
    return parameters_to_vector(params).detach()


def _unpack_params(theta_t, models):
    params = []
    for m in models:
        params += list(m.parameters())
    vector_to_parameters(theta_t, params)


def save_models(cfg, U_theta, S_phi, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "U_state_dict": U_theta.state_dict(),
        "S_state_dict": S_phi.state_dict(),
        "cfg": asdict(cfg),
    }, path)


def load_models(cfg, U_theta, S_phi, path):
    blob = torch.load(path, map_location=cfg.device, weights_only=False)
    U_theta.load_state_dict(blob["U_state_dict"])
    S_phi.load_state_dict(blob["S_state_dict"])


def eval_S_error(S_phi, x_grid, device):
    """Relative L2 error of S_phi(x) vs true_force(x) on the grid."""
    with torch.no_grad():
        x_t = torch.tensor(x_grid, dtype=torch.float64, device=device).view(-1, 1)
        y_pred = S_phi(x_t).cpu().numpy().ravel()
    y_true = true_force(x_grid)
    return float(rel_l2_error(y_pred, y_true))


# ================================================================
# Training: Adam warmup + SciPy BFGS (SSBroyden2)
# ================================================================

def train_pinn(u0: np.ndarray, uT_target: np.ndarray, cfg: BurgersConfig) -> Dict[str, Any]:
    """
    Train PINN with U_theta(x,t) and S_phi(x).
    Direct residual loss only. Tracks rel L2 error during training.
    """
    device = cfg.device
    N = cfg.N
    x = np.linspace(0, cfg.L, N, endpoint=False, dtype=np.float64)

    x_grid_t = torch.tensor(x, dtype=torch.float64, device=device).reshape(-1, 1)
    x_T_t = x_grid_t.clone()
    u0_t = torch.tensor(u0, dtype=torch.float64, device=device).reshape(-1, 1)
    uT_t = torch.tensor(uT_target, dtype=torch.float64, device=device).reshape(-1, 1)

    # Models
    U_theta = MLP(2, 1, cfg.hidden_U, cfg.layers_U, cfg.act_U).to(device).double()
    S_phi = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(device).double()

    batch = _make_fixed_batch(cfg, device, x_grid_t, u0_t, x_T_t, uT_t)

    # ---- Adam warmup ----
    history = {
        "adam": {"loss": [], "L_res": [], "L_ic": [], "L_bc": [], "L_data": [],
                 "rel_l2_uT": [], "rel_l2_f": [], "lr": []},
        "bfgs": {"loss": [], "grad_norm": [], "L_res": [], "L_ic": [], "L_bc": [], "L_data": [],
                 "rel_l2_uT": [], "rel_l2_f": []},
    }

    def _compute_rel_errors():
        with torch.no_grad():
            xt_T = torch.cat([x_T_t, torch.full_like(x_T_t, cfg.t_final)], dim=1)
            u_T_pred = U_theta(xt_T).cpu().numpy().ravel()
        rel_uT = float(rel_l2_error(u_T_pred, uT_target))
        rel_f = eval_S_error(S_phi, x, device)
        return rel_uT, rel_f

    t0 = time.perf_counter()
    warmup_steps = max(0, int(cfg.scipy_pinn_adam_warmup_steps))
    if warmup_steps > 0:
        param_list = list(U_theta.parameters()) + list(S_phi.parameters())
        opt = torch.optim.Adam(param_list, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, warmup_steps // 5), gamma=0.3)
        for it in range(1, warmup_steps + 1):
            opt.zero_grad(set_to_none=True)
            L, Ldict = _pinn_loss(cfg, U_theta, S_phi, batch)
            L.backward()
            opt.step()
            scheduler.step()

            loss_val = float(L.detach())
            history["adam"]["loss"].append(loss_val)
            for key in ["L_res", "L_ic", "L_bc", "L_data"]:
                history["adam"][key].append(Ldict[key])
            history["adam"]["lr"].append(scheduler.get_last_lr()[0])

            rel_uT, rel_f = _compute_rel_errors()
            history["adam"]["rel_l2_uT"].append(rel_uT)
            history["adam"]["rel_l2_f"].append(rel_f)

            if it == 1 or it % 200 == 0 or it == warmup_steps:
                print(f"[Adam {it:4d}/{warmup_steps}] loss={loss_val:.3e} "
                      f"res={Ldict['L_res']:.2e} ic={Ldict['L_ic']:.2e} "
                      f"bc={Ldict['L_bc']:.2e} data={Ldict['L_data']:.2e} "
                      f"rel_uT={rel_uT:.3e} rel_f={rel_f:.3e}")

    # ---- SciPy BFGS ----
    with torch.no_grad():
        theta0_t = _pack_params((U_theta, S_phi))
    theta0 = theta0_t.cpu().numpy()

    eval_hist = []
    iter_state = {"eval_start": 0, "k": 0}
    last_eval = {}

    def fun_and_jac(theta_np):
        theta_t = torch.tensor(theta_np, dtype=torch.float64, device=device)
        _unpack_params(theta_t, (U_theta, S_phi))

        for p in list(U_theta.parameters()) + list(S_phi.parameters()):
            if p.grad is not None:
                p.grad = None

        L, Ldict = _pinn_loss(cfg, U_theta, S_phi, batch)
        L.backward()

        grads = [torch.zeros_like(p) if p.grad is None else p.grad
                 for p in list(U_theta.parameters()) + list(S_phi.parameters())]
        gvec = parameters_to_vector(grads)

        loss_val = float(L.detach())
        gnorm = float(torch.linalg.norm(gvec).detach())

        rel_uT, rel_f = _compute_rel_errors()

        entry = {"loss": loss_val, "grad_norm": gnorm,
                 "rel_l2_uT": rel_uT, "rel_l2_f": rel_f, **Ldict}
        eval_hist.append(entry)
        last_eval.update(entry)

        return loss_val, gvec.detach().cpu().numpy()

    def cb(_xk):
        start = iter_state["eval_start"]
        end = len(eval_hist)
        last = eval_hist[end - 1] if end > start else last_eval
        for key in ["loss", "grad_norm", "L_res", "L_ic", "L_bc", "L_data",
                     "rel_l2_uT", "rel_l2_f"]:
            history["bfgs"][key].append(last.get(key, 0.0))
        iter_state["eval_start"] = end
        iter_state["k"] += 1

        k = iter_state["k"]
        if k % 10 == 0 or k == 1:
            print(f"[BFGS iter {k:4d}] loss={last['loss']:.3e} |g|={last['grad_norm']:.2e} "
                  f"rel_uT={last['rel_l2_uT']:.3e} rel_f={last['rel_l2_f']:.3e}")

    H0 = np.eye(len(theta0))
    for ep in range(1, cfg.scipy_pinn_epochs + 1):
        theta0_t = torch.tensor(theta0, dtype=torch.float64, device=device)
        _unpack_params(theta0_t, (U_theta, S_phi))

        print(f"\n[Epoch {ep}/{cfg.scipy_pinn_epochs}] rel_f = {eval_S_error(S_phi, x, device):.3e}")

        res = minimize(
            fun=fun_and_jac, x0=theta0, jac=True, method="BFGS",
            tol=cfg.scipy_ftol, callback=cb,
            options=dict(
                maxiter=cfg.scipy_pinn_maxiter,
                gtol=cfg.scipy_gtol,
                disp=cfg.scipy_disp,
                method_bfgs=cfg.scipy_method_bfgs,
                hess_inv0=H0, initial_scale=False,
            ),
        )
        theta0 = res.x.copy()

        H_new = getattr(res, "hess_inv", None)
        if H_new is not None:
            H0 = 0.5 * (H_new + H_new.T)
            try:
                cholesky(H0)
            except LinAlgError:
                H0 = np.eye(len(theta0))
        else:
            H0 = np.eye(len(theta0))

        # Resample for next epoch
        batch = _make_fixed_batch(cfg, device, x_grid_t, u0_t, x_T_t, uT_t, seed=ep)

    t1 = time.perf_counter()
    history["runtime_sec"] = t1 - t0

    # Save
    os.makedirs(os.path.dirname(cfg.pinn_scipy_path), exist_ok=True)
    for phase in ["adam", "bfgs"]:
        for key in history[phase]:
            history[phase][key] = np.asarray(history[phase][key], dtype=float)

    np.savez_compressed(
        cfg.pinn_scipy_path,
        **{f"adam_{k}": v for k, v in history["adam"].items()},
        **{f"bfgs_{k}": v for k, v in history["bfgs"].items()},
        runtime_sec=np.array(history["runtime_sec"]),
        cfg_snapshot=np.array(asdict(cfg), dtype=object),
    )
    save_models(cfg, U_theta, S_phi, cfg.pinn_scipy_model)
    print(f"Saved PINN results to {cfg.pinn_scipy_path}")

    return history
