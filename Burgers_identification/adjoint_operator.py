"""
Discrete adjoint of Crank-Nicolson Burgers with NN forcing s(x) = S_phi(x).

PDE:  u_t + u*u_x - nu*u_xx = s(x),  periodic BCs.

Provides:
  - nn_force_forward / nn_force_backward  (NN force helpers)
  - BurgersAdjointNN                      (forward + adjoint sweep through CN)
  - invert_force_adjoint                  (full training loop)
  - gradient_check                        (finite-difference vs adjoint)
  - simulate_with_phi                     (re-simulate with learned phi)
"""

import os
import time
import numpy as np
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from scipy.optimize import minimize
from numpy.linalg import cholesky, LinAlgError

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from cfg import BurgersConfig
from PINN import MLP
from solver import make_D_L, BurgersFDCore, rel_l2_error, true_force

torch.set_default_dtype(torch.float64)


# ================================================================
# NN force helpers
# ================================================================

def _assign_params(model: nn.Module, phi: np.ndarray):
    dev = next(model.parameters()).device
    vector_to_parameters(
        torch.tensor(phi, dtype=torch.float64, device=dev),
        list(model.parameters()),
    )


def _flatten_params(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return parameters_to_vector(list(model.parameters())).cpu().numpy().astype(float)


def get_init_phi(cfg: BurgersConfig) -> np.ndarray:
    """Return a random initial parameter vector for the force NN."""
    m = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(cfg.device).double()
    return _flatten_params(m)


def nn_force_forward(x_np, phi, cfg):
    """Evaluate s = S_phi(x) on grid x_np. Returns (s, cache)."""
    x_np = np.asarray(x_np, dtype=float).ravel()
    N = x_np.size
    model = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(cfg.device).double()
    _assign_params(model, phi)

    x_t = torch.tensor(x_np, dtype=torch.float64, device=cfg.device).view(-1, 1)
    with torch.no_grad():
        s_t = model(x_t)
    s = s_t.cpu().numpy().reshape(N)
    cache = {"x": x_np.copy(), "phi": phi.copy()}
    return s, cache


def nn_force_backward(adj_s, cache, phi, cfg):
    """VJP: dL/ds -> dL/dphi via backprop through NN."""
    phi_vec = np.asarray(phi, dtype=float).ravel()
    model = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(cfg.device).double()
    _assign_params(model, phi_vec)

    x_np = np.asarray(cache["x"], dtype=float).ravel()
    N = x_np.size
    x_t = torch.tensor(x_np, dtype=torch.float64, device=cfg.device).view(-1, 1)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
        p.requires_grad_(True)

    s_t = model(x_t)
    adj = torch.tensor(adj_s.reshape(N, 1), dtype=torch.float64, device=cfg.device)
    s_t.backward(gradient=adj)

    grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in model.parameters()]
    return parameters_to_vector(grads).detach().cpu().numpy().astype(float)


# ================================================================
# Discrete adjoint class (through Crank-Nicolson)
# ================================================================

class BurgersAdjointNN:
    """Discrete adjoint through CN Burgers with NN forcing s(x) = S_phi(x)."""

    def __init__(self, cfg: BurgersConfig):
        self.cfg = cfg
        self.core = BurgersFDCore(cfg)

    def _Jf(self, u):
        """Jacobian of f(u,s) wrt u: -diag(Du) - diag(u)@D + nu*L"""
        D, Lx, nu = self.core.D, self.core.Lx, self.core.nu
        Du = D @ u
        return -np.diag(Du) - np.diag(u) @ D + nu * Lx

    def forward(self, u0, phi):
        """Forward CN solve with NN force, storing trajectory for adjoint."""
        core = self.core
        s, cache_s = nn_force_forward(core.x, phi, self.cfg)
        fwd = core.forward(u0, s, store_trajectory=True)
        return fwd["u_final"], fwd["U_hist"], s, cache_s

    def scipy_objective(self, phi, u0, uT_target):
        """Returns (loss, grad_phi). Loss = 0.5 * dx * ||u(T) - uT_target||^2 + reg."""
        core = self.core
        dt, dx, Nt, N = core.dt, core.dx, core.Nt, core.N
        alpha = self.cfg.alpha_reg

        # Forward solve
        uT, U_hist, s_np, cache_s = self.forward(u0, phi)

        # Objective
        mis = uT - uT_target
        J_data = 0.5 * dx * np.dot(mis, mis)
        J_reg = 0.5 * alpha * np.dot(phi, phi)
        loss = J_data + J_reg

        # Build CN Jacobians along trajectory
        I = np.eye(N)
        A_list, B_list = [], []
        for n in range(Nt):
            Jf_n = self._Jf(U_hist[n])
            Jf_np1 = self._Jf(U_hist[n + 1])
            A_list.append(-I - 0.5 * dt * Jf_n)
            B_list.append(I - 0.5 * dt * Jf_np1)

        # Discrete adjoint backward sweep
        dJ_duT = dx * mis
        lam_list = [None] * (Nt + 1)
        lam_list[Nt] = np.linalg.solve(B_list[-1].T, -dJ_duT)

        for n in range(Nt - 1, 0, -1):
            rhs = -A_list[n].T @ lam_list[n + 1]
            lam_list[n] = np.linalg.solve(B_list[n - 1].T, rhs)

        rhs0 = -A_list[0].T @ lam_list[1]
        lam_list[0] = np.linalg.solve(B_list[0].T, rhs0)

        # Gradient wrt s: dR/ds = -dt*I => dJ/ds = -dt * sum(lambda)
        lam_sum = np.zeros(N)
        for n in range(Nt):
            lam_sum += lam_list[n + 1]

        # VJP through NN: (ds/dphi)^T * (-dt * lam_sum)
        grad_phi = alpha * phi + nn_force_backward(-dt * lam_sum, cache_s, phi, self.cfg)

        return loss, grad_phi


# ================================================================
# Gradient check (finite difference vs adjoint)
# ================================================================

def gradient_check(cfg, u0, uT_target, phi, n_checks=10, eps_fd=1e-5):
    adj = BurgersAdjointNN(cfg)
    loss0, grad_adj = adj.scipy_objective(phi, u0, uT_target)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(phi), size=min(n_checks, len(phi)), replace=False)

    results = []
    for i in indices:
        phi_p = phi.copy(); phi_p[i] += eps_fd
        phi_m = phi.copy(); phi_m[i] -= eps_fd
        fp, _ = adj.scipy_objective(phi_p, u0, uT_target)
        fm, _ = adj.scipy_objective(phi_m, u0, uT_target)
        fd = (fp - fm) / (2 * eps_fd)
        adj_val = grad_adj[i]
        rel_err = abs(fd - adj_val) / (abs(adj_val) + 1e-30)
        results.append({"idx": int(i), "fd": fd, "adj": adj_val, "rel_err": rel_err})

    print(f"{'idx':>5s} {'FD grad':>14s} {'Adj grad':>14s} {'rel err':>12s}")
    for r in results:
        print(f"{r['idx']:5d} {r['fd']:14.6e} {r['adj']:14.6e} {r['rel_err']:12.4e}")

    max_err = max(r["rel_err"] for r in results)
    print(f"\nMax relative error: {max_err:.4e}")
    return results


# ================================================================
# Adam warmup
# ================================================================

def adam_warmup(phi0, fun, n_steps, lr0=1e-3):
    phi = phi0.copy()
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step_size = max(1, n_steps // 10)
    gamma = 0.6
    m = np.zeros_like(phi)
    v = np.zeros_like(phi)

    for k in range(1, n_steps + 1):
        f, g = fun(phi)
        lr_k = lr0 * (gamma ** ((k - 1) // step_size))
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        phi -= lr_k * m_hat / (np.sqrt(v_hat) + eps)

    return phi


# ================================================================
# Main training loop
# ================================================================

def invert_force_adjoint(u0, uT_target, cfg, phi0, n_steps_warmup=0):
    """
    Train NN s(x) = S_phi(x) via discrete adjoint + SSBroyden2 BFGS.
    Tracks relative L2 error of terminal solution and force during training.
    """
    adj = BurgersAdjointNN(cfg)
    x_grid = adj.core.x
    s_true = true_force(x_grid)

    history = {"evals": [], "iters": []}
    k_eval = {"count": 0}
    last_eval = {"loss": None, "grad_norm": None}

    def fun_with_hist(ph):
        loss, grad = adj.scipy_objective(ph, u0, uT_target)
        gnorm = float(np.linalg.norm(grad))

        # Track terminal solution error
        uT_pred, _, _, _ = adj.forward(u0, ph)
        rel_l2_T = float(rel_l2_error(uT_pred, uT_target))

        # Track force error
        s_pred, _ = nn_force_forward(x_grid, ph, cfg)
        rel_l2_f = float(rel_l2_error(s_pred, s_true))

        k = k_eval["count"]
        history["evals"].append({
            "k": k, "loss": float(loss), "grad_norm": gnorm,
            "rel_l2_uT": rel_l2_T, "rel_l2_f": rel_l2_f,
        })
        last_eval.update(history["evals"][-1])
        k_eval["count"] += 1
        if cfg.scipy_verbose:
            print(f"[Adj] feval={k:04d} f={loss:.6e} |g|={gnorm:.3e} "
                  f"rel_uT={rel_l2_T:.3e} rel_f={rel_l2_f:.3e}")
        return loss, grad

    def cb(phi_k):
        k = len(history["iters"])
        history["iters"].append({"k": k, **last_eval})
        if k % 10 == 0 or k == 0:
            e = last_eval
            print(f"  iter {k:04d}: loss={e['loss']:.6e} |g|={e['grad_norm']:.3e} "
                  f"rel_uT={e['rel_l2_uT']:.3e} rel_f={e['rel_l2_f']:.3e}")

    t0 = time.perf_counter()

    # Adam warmup
    if n_steps_warmup > 0:
        phi0 = adam_warmup(phi0, fun_with_hist, n_steps_warmup, lr0=cfg.GD_warmup_lr)
        # Reset for BFGS phase
        history = {"evals": [], "iters": []}
        k_eval["count"] = 0

    # BFGS
    print("Starting BFGS optimization...")
    H0 = np.eye(len(phi0))
    res = minimize(
        fun=fun_with_hist, x0=phi0, jac=True, method="BFGS",
        callback=cb, tol=cfg.scipy_ftol,
        options=dict(
            maxiter=cfg.scipy_adj_nn_maxiter,
            gtol=cfg.scipy_gtol,
            disp=cfg.scipy_disp,
            method_bfgs=cfg.scipy_method_bfgs,
            hess_inv0=H0, initial_scale=False,
        ),
    )

    t1 = time.perf_counter()
    history["runtime_sec"] = t1 - t0

    phi_opt = res.x.astype(float)
    s_opt, _ = nn_force_forward(x_grid, phi_opt, cfg)

    # Save
    os.makedirs(os.path.dirname(cfg.adj_nn_path), exist_ok=True)
    eval_loss = np.array([e["loss"] for e in history["evals"]])
    eval_gnorm = np.array([e["grad_norm"] for e in history["evals"]])
    eval_rel_uT = np.array([e["rel_l2_uT"] for e in history["evals"]])
    eval_rel_f = np.array([e["rel_l2_f"] for e in history["evals"]])

    iter_loss = np.array([i["loss"] for i in history["iters"]]) if history["iters"] else np.array([])
    iter_gnorm = np.array([i["grad_norm"] for i in history["iters"]]) if history["iters"] else np.array([])
    iter_rel_uT = np.array([i["rel_l2_uT"] for i in history["iters"]]) if history["iters"] else np.array([])
    iter_rel_f = np.array([i["rel_l2_f"] for i in history["iters"]]) if history["iters"] else np.array([])

    np.savez_compressed(
        cfg.adj_nn_path,
        phi_opt=phi_opt, force_opt=s_opt,
        eval_loss=eval_loss, eval_grad_norm=eval_gnorm,
        eval_rel_l2_uT=eval_rel_uT, eval_rel_l2_f=eval_rel_f,
        iter_loss=iter_loss, iter_grad_norm=iter_gnorm,
        iter_rel_l2_uT=iter_rel_uT, iter_rel_l2_f=iter_rel_f,
        runtime_sec=np.array(history["runtime_sec"]),
        cfg_snapshot=asdict(cfg),
    )
    print(f"Saved adjoint results to {cfg.adj_nn_path}")
    return phi_opt, history


# ================================================================
# Simulate with learned phi
# ================================================================

def simulate_with_phi(cfg, u0, phi):
    """Re-simulate Burgers CN with learned NN forcing. Returns (u_final, s_pred, s_true)."""
    core = BurgersFDCore(cfg)
    s_pred, _ = nn_force_forward(core.x, phi, cfg)
    s_true_val = true_force(core.x)
    fwd = core.forward(u0, s_pred, store_trajectory=True)
    return fwd["u_final"], fwd["U_hist"], s_pred, s_true_val
