import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError, cholesky
from scipy.optimize import minimize

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from cfg import AllenCahn3DConfig
from PINN import MLP
from solver import _get_linear_solver, make_solver_cache, reaction_true, rel_l2_error, solve_allen_cahn_imex

torch.set_default_dtype(torch.float64)


def _assign_params(model: nn.Module, phi: np.ndarray):
    dev = next(model.parameters()).device
    vector_to_parameters(torch.tensor(phi, dtype=torch.float64, device=dev), list(model.parameters()))


def _flatten_params(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return parameters_to_vector(list(model.parameters())).detach().cpu().numpy().astype(float)


def get_init_phi(cfg: AllenCahn3DConfig) -> np.ndarray:
    model = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(cfg.device).double()
    return _flatten_params(model)


def nn_force_forward(u, phi, cfg: AllenCahn3DConfig):
    u_np = np.asarray(u, dtype=float).reshape(-1)
    model = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(cfg.device).double()
    _assign_params(model, phi)

    u_t = torch.tensor(u_np, dtype=torch.float64, device=cfg.device, requires_grad=True).view(-1, 1)
    f_t = model(u_t)
    df_du_t = torch.autograd.grad(f_t, u_t, torch.ones_like(f_t), create_graph=False)[0]

    f = f_t.detach().cpu().numpy().reshape(-1)
    df_du = df_du_t.detach().cpu().numpy().reshape(-1)
    cache = {"u": u_np.copy(), "phi": np.asarray(phi, dtype=float).copy()}
    return f, cache, df_du


def nn_force_backward(adj_f, cache, phi, cfg: AllenCahn3DConfig):
    phi_vec = np.asarray(phi, dtype=float).reshape(-1)
    model = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(cfg.device).double()
    _assign_params(model, phi_vec)

    u_np = np.asarray(cache["u"], dtype=float).reshape(-1)
    u_t = torch.tensor(u_np, dtype=torch.float64, device=cfg.device).view(-1, 1)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
        param.requires_grad_(True)

    f_t = model(u_t)
    adj = torch.tensor(np.asarray(adj_f, dtype=float).reshape(-1, 1), dtype=torch.float64, device=cfg.device)
    f_t.backward(gradient=adj)

    grads = [torch.zeros_like(param) if param.grad is None else param.grad for param in model.parameters()]
    return parameters_to_vector(grads).detach().cpu().numpy().astype(float)


def make_reaction_function(phi, cfg: AllenCahn3DConfig):
    def reaction(u_field):
        f, _, _ = nn_force_forward(np.asarray(u_field, dtype=float).reshape(-1), phi, cfg)
        return f.reshape(np.asarray(u_field).shape)

    return reaction


class AllenCahnIMEXAdjointNN:
    def __init__(self, cfg: AllenCahn3DConfig):
        self.cfg = cfg
        self.solver_cache = make_solver_cache(cfg)
        self.cell_volume = self.solver_cache["hx"] * self.solver_cache["hy"] * self.solver_cache["hz"]

    def forward(self, u0, phi):
        cfg = self.cfg
        t = 0.0
        dt_full = float(cfg.dt)
        tol_dt = 1.0e-14 * max(1.0, abs(dt_full))

        u_curr = np.asarray(u0, dtype=float).reshape(-1).copy()
        f_curr, cache_curr, dfdu_curr = nn_force_forward(u_curr, phi, cfg)
        u_prev = None
        f_prev = None
        cache_prev = None
        dfdu_prev = None
        prev_dt = None
        history = []

        while t < cfg.t_final - 1.0e-14:
            dt = min(cfg.dt, cfg.t_final - t)
            use_bdf2 = (
                u_prev is not None
                and prev_dt is not None
                and abs(dt - prev_dt) <= tol_dt
                and abs(dt - dt_full) <= tol_dt
            )

            if use_bdf2:
                solve_linear = _get_linear_solver(self.solver_cache, cfg, dt, mass_coeff=3.0, diff_coeff=2.0)
                rhs = 4.0 * u_curr - u_prev + 2.0 * dt * (2.0 * f_curr - f_prev)
                history.append(
                    {
                        "scheme": "bdf2",
                        "dt": dt,
                        "cache_prev": cache_prev,
                        "dfdu_prev": dfdu_prev,
                        "cache_curr": cache_curr,
                        "dfdu_curr": dfdu_curr,
                    }
                )
            else:
                solve_linear = _get_linear_solver(self.solver_cache, cfg, dt)
                rhs = u_curr + dt * f_curr
                history.append({"scheme": "euler", "dt": dt, "cache_curr": cache_curr, "dfdu_curr": dfdu_curr})

            u_new = np.asarray(solve_linear(rhs), dtype=float).reshape(-1)
            t += dt

            u_prev = u_curr
            f_prev = f_curr
            cache_prev = cache_curr
            dfdu_prev = dfdu_curr
            prev_dt = dt
            u_curr = u_new
            if t < cfg.t_final - 1.0e-14:
                f_curr, cache_curr, dfdu_curr = nn_force_forward(u_curr, phi, cfg)

        return u_curr.reshape(cfg.Nx, cfg.Ny, cfg.Nz), history

    def scipy_objective(self, phi, u0, uT_target):
        cfg = self.cfg
        alpha = cfg.alpha_reg
        volume = self.cell_volume

        uT, history = self.forward(u0, phi)
        diff = np.asarray(uT - uT_target, dtype=float).reshape(-1)
        loss = 0.5 * volume * np.sum(diff**2)

        uT_flat = np.asarray(uT, dtype=float).reshape(-1)
        f_T, cache_T, dfdu_T = nn_force_forward(uT_flat, phi, cfg)
        reg_loss = 0.5 * alpha * volume * np.sum(f_T**2)
        loss += reg_loss

        n_state = diff.size
        state_loads = [np.zeros(n_state, dtype=float) for _ in range(len(history) + 1)]
        state_loads[-1] = volume * diff + alpha * volume * (f_T * dfdu_T)
        grad_phi = np.zeros_like(phi)

        if alpha != 0.0:
            grad_phi += nn_force_backward(alpha * volume * f_T, cache_T, phi, cfg)

        for idx in range(len(history) - 1, -1, -1):
            step = history[idx]
            dt = step["dt"]
            lam_out = state_loads[idx + 1]

            if step["scheme"] == "bdf2":
                solve_linear = _get_linear_solver(self.solver_cache, cfg, dt, mass_coeff=3.0, diff_coeff=2.0)
                q = np.asarray(solve_linear(lam_out), dtype=float).reshape(-1)
                grad_phi += 4.0 * dt * nn_force_backward(q, step["cache_curr"], phi, cfg)
                grad_phi -= 2.0 * dt * nn_force_backward(q, step["cache_prev"], phi, cfg)
                state_loads[idx] += 4.0 * q + 4.0 * dt * step["dfdu_curr"] * q
                state_loads[idx - 1] += -q - 2.0 * dt * step["dfdu_prev"] * q
            else:
                solve_linear = _get_linear_solver(self.solver_cache, cfg, dt)
                q = np.asarray(solve_linear(lam_out), dtype=float).reshape(-1)
                grad_phi += dt * nn_force_backward(q, step["cache_curr"], phi, cfg)
                state_loads[idx] += q + dt * step["dfdu_curr"] * q

        return loss, grad_phi


def gradient_check(
    cfg: AllenCahn3DConfig,
    u0: np.ndarray,
    uT_target: np.ndarray,
    phi: np.ndarray,
    n_dirs: int = 6,
    eps_fd: float = 1.0e-6,
):
    adj = AllenCahnIMEXAdjointNN(cfg)
    loss0, grad_adj = adj.scipy_objective(phi, u0, uT_target)

    rng = np.random.default_rng(42)
    rows = []
    for idx in range(n_dirs):
        direction = rng.standard_normal(phi.shape)
        direction /= max(np.linalg.norm(direction), 1.0e-30)
        dir_adj = float(np.dot(grad_adj, direction))
        f_plus, _ = adj.scipy_objective(phi + eps_fd * direction, u0, uT_target)
        f_minus, _ = adj.scipy_objective(phi - eps_fd * direction, u0, uT_target)
        dir_fd = (f_plus - f_minus) / (2.0 * eps_fd)
        rel_err = abs(dir_adj - dir_fd) / (abs(dir_adj) + 1.0e-30)
        rows.append({"dir": idx, "adj": dir_adj, "fd": dir_fd, "rel_err": rel_err})

    return {
        "loss": float(loss0),
        "grad_norm": float(np.linalg.norm(grad_adj)),
        "rows": rows,
        "max_rel_err": max(row["rel_err"] for row in rows),
        "median_rel_err": float(np.median([row["rel_err"] for row in rows])),
    }


def adam_warmup(phi0, fun, cfg: AllenCahn3DConfig, n_steps: int, cb=None):
    phi = phi0.copy()
    print_every = 100
    lr0 = cfg.GD_warmup_lr
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    step_size = max(1, n_steps // 10)
    gamma = 0.6
    m = np.zeros_like(phi)
    v = np.zeros_like(phi)

    for k in range(1, n_steps + 1):
        f_val, grad = fun(phi)
        lr_k = lr0 * (gamma ** ((k - 1) // step_size))
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1**k)
        v_hat = v / (1.0 - beta2**k)
        phi -= lr_k * m_hat / (np.sqrt(v_hat) + eps)
        if cb is not None and (k == 1 or k % print_every == 0 or k == n_steps):
            cb(k, phi, f_val, grad)

    return phi


def invert_force_adjoint(
    u0: np.ndarray,
    uT_target: np.ndarray,
    cfg: AllenCahn3DConfig,
    phi0: np.ndarray,
    n_steps_warmup: int = 40,
    u_ref_hist: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    adj = AllenCahnIMEXAdjointNN(cfg)
    history: Dict[str, Any] = {"evals": [], "iters": []}
    eval_count = {"k": 0}
    last_eval: Dict[str, float] = {}

    def fun_with_hist(phi):
        loss, grad = adj.scipy_objective(phi, u0, uT_target)
        grad_norm = float(np.linalg.norm(grad))
        uT_pred, _ = adj.forward(u0, phi)
        rel_uT = float(rel_l2_error(uT_pred, uT_target))

        u_test = np.linspace(-1.0, 1.0, 1001)
        f_pred, _, _ = nn_force_forward(u_test, phi, cfg)
        f_true = reaction_true(u_test)
        rel_f = float(rel_l2_error(f_pred, f_true))

        row = {
            "k": eval_count["k"],
            "loss": float(loss),
            "grad_norm": grad_norm,
            "rel_l2_uT": rel_uT,
            "rel_l2_f": rel_f,
        }
        history["evals"].append(row)
        last_eval.update(row)
        eval_count["k"] += 1
        if cfg.scipy_verbose:
            print(
                f"[Adj] feval={row['k']:04d} f={row['loss']:.6e} |g|={row['grad_norm']:.3e} "
                f"rel_uT={row['rel_l2_uT']:.3e} rel_f={row['rel_l2_f']:.3e}"
            )
        return loss, grad

    def warmup_callback(step, phi, loss, grad):
        uT_pred, _ = adj.forward(u0, phi)
        row = {
            "phase": "adam",
            "step": int(step),
            "loss": float(loss),
            "grad_norm": float(np.linalg.norm(grad)),
            "rel_l2_uT": float(rel_l2_error(uT_pred, uT_target)),
        }
        history["iters"].append(row)
        print(
            f"[Adj][adam] step={row['step']:04d} loss={row['loss']:.6e} "
            f"|g|={row['grad_norm']:.3e} rel_uT={row['rel_l2_uT']:.3e}"
        )

    t0 = time.perf_counter()
    phi = phi0.copy()
    if n_steps_warmup > 0:
        phi = adam_warmup(phi, fun_with_hist, cfg, n_steps_warmup, cb=warmup_callback)

    H0 = np.eye(len(phi))
    iter_state = {"eval_start": 0, "bfgs_iter": 0}

    def cb(_phi):
        start = iter_state["eval_start"]
        end = len(history["evals"])
        row = history["evals"][end - 1] if end > start else last_eval
        iter_state["bfgs_iter"] += 1
        iter_row = {
            "phase": "bfgs",
            "iter": iter_state["bfgs_iter"],
            **row,
        }
        history["iters"].append(iter_row)
        iter_state["eval_start"] = end
        if iter_row["iter"] == 1 or iter_row["iter"] % 20 == 0:
            print(
                f"[Adj][bfgs] iter={iter_row['iter']:04d} "
                f"loss={iter_row['loss']:.6e} |g|={iter_row['grad_norm']:.3e} "
                f"rel_uT={iter_row['rel_l2_uT']:.3e} rel_f={iter_row['rel_l2_f']:.3e}"
            )

    result = minimize(
        fun=fun_with_hist,
        x0=phi,
        jac=True,
        method="BFGS",
        callback=cb,
        tol=cfg.scipy_ftol,
        options=dict(
            maxiter=cfg.scipy_adj_nn_maxiter,
            maxfun=cfg.scipy_adj_nn_maxfun,
            gtol=cfg.scipy_gtol,
            disp=cfg.scipy_disp,
            method_bfgs=cfg.scipy_method_bfgs,
            hess_inv0=H0,
            initial_scale=False,
        ),
    )
    phi = result.x.copy()
    if getattr(result, "hess_inv", None) is not None:
        H0 = 0.5 * (result.hess_inv + result.hess_inv.T)
        try:
            cholesky(H0)
        except LinAlgError:
            H0 = np.eye(len(phi))
    else:
        H0 = np.eye(len(phi))

    print(
        f"[Adj][bfgs] done status={result.status} "
        f"loss={float(result.fun):.6e} nfev={getattr(result, 'nfev', -1)}"
    )

    t1 = time.perf_counter()
    history["runtime_sec"] = t1 - t0

    os.makedirs(os.path.dirname(cfg.adj_nn_path), exist_ok=True)
    np.savez_compressed(
        cfg.adj_nn_path,
        evals=np.array(history["evals"], dtype=object),
        iters=np.array(history["iters"], dtype=object),
        phi=phi,
        runtime_sec=np.array(history["runtime_sec"]),
        cfg_snapshot=np.array(asdict(cfg), dtype=object),
    )
    torch.save({"phi": phi, "cfg": asdict(cfg)}, cfg.adj_nn_model)

    return phi, history


def simulate_trajectory_with_phi(cfg: AllenCahn3DConfig, u0: np.ndarray, phi: np.ndarray):
    reaction = make_reaction_function(phi, cfg)
    u_hist, t_hist, _ = solve_allen_cahn_imex(cfg, u0, reaction_fun=reaction)
    return u_hist, t_hist
