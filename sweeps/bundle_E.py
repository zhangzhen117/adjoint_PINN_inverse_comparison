"""Bundle E -- noise sensitivity on Darcy (referee R3.6).

R3.6 notes that only the Darcy benchmark uses noisy observations and asks for a
short study varying the noise level. Darcy carries it because it is by far the
cheapest benchmark (the adjoint inverts in ~3 s) and because its observation model
already supports a noise level, type and seed.

Two sub-sweeps:

* **E1** -- sigma in {0, 0.1%, 1%, 5%, 10%} x 5 seeds x {adjoint, PINN, EKI}.
* **E2** -- gamma in {0, 1e-4, 1e-3, 1e-2} at sigma in {1%, 5%} x 5 seeds x
  {adjoint, PINN}. Without a regularization axis, the degradation reported at
  sigma = 10% would partly be an artifact of holding gamma at a value chosen for
  1% noise, since the optimal smoothing grows with the noise level. E2 lets the
  paper either show the ranking is stable in gamma or report each method at its
  own best gamma.

One ``seed`` drives both the noise realization and the network initialization, so
each seed is an independent replication of the entire pipeline rather than of one
stage of it. The adjoint and EKI share the KL parameterization; the PINN uses a
neural field, with its H1 weight converted by the truth-free calibration in
``cases/case2_darcy/cfg.py`` (the notebook derived that constant from ``m_true``).

This sweep also re-establishes the Test 2 headline numbers, which could not be
reproduced from the committed history files.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cases", "case2_darcy"))

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
NOISE_LEVELS = (0.0, 0.001, 0.01, 0.05, 0.10)
GAMMAS = (0.0, 1e-4, 1e-3, 1e-2)
GAMMA_NOISE = (0.01, 0.05)
BASE_GAMMA = 1e-3


def rows():
    out = []
    # E1: noise sweep at the production gamma.
    for noise in NOISE_LEVELS:
        for method in ("adjoint", "pinn", "eki"):
            for seed in SEEDS:
                out.append({"sub": "E1", "method": method, "noise": noise,
                            "gamma": BASE_GAMMA, "seed": seed})
    # E2: gamma x noise, gradient-based methods only.
    for noise in GAMMA_NOISE:
        for gamma in GAMMAS:
            if gamma == BASE_GAMMA:
                continue          # already covered by E1 at this noise level
            for method in ("adjoint", "pinn"):
                for seed in SEEDS:
                    out.append({"sub": "E2", "method": method, "noise": noise,
                                "gamma": gamma, "seed": seed})
    return out


def _setup(cfg, seed):
    """Build the FEM inverse operator, the truth, and the (noisy) observations."""
    from darcy_adjoint import DarcyInverse, create_observation_grid
    from GRF import GaussianRF

    obs_points = create_observation_grid(n_x=cfg.n_obs_x, n_y=cfg.n_obs_y,
                                         margin=cfg.obs_margin)
    inv = DarcyInverse(
        nx=cfg.nx, ny=cfg.ny, element_degree=cfg.element_degree,
        f_given=lambda x, y: cfg.source_value * np.ones_like(x),
        beta=cfg.beta, gamma=cfg.gamma,
        solver_type=cfg.linear_solver, obs_points=obs_points,
    )
    RF = GaussianRF(cfg.grf_tau, cfg.grf_alpha, cfg.grf_sigma)

    # The truth is fixed across seeds: seeds replicate the noise and the
    # initialization, not the inverse problem itself.
    coefs_true = np.random.RandomState(cfg.truth_seed).randn(cfg.grf_modes)
    m_true = RF.sample(inv.element_centroids, coefs_true).flatten()

    y_obs, u_clean = inv.generate_observations(
        m_true, noise_level=cfg.noise_level, noise_type=cfg.noise_type,
        seed=cfg.noise_seed)
    return inv, RF, coefs_true, m_true, y_obs, u_clean, obs_points


def _kl_basis(inv, RF, cfg):
    """Scaled KL eigenfunction matrix: ``m = scaled_Phi.T @ xi``."""
    sigma = RF.sigma if RF.sigma is not None else RF.tau ** (0.5 * (2 * RF.alpha - 2))
    sqrt_eigs, kl_points = RF._eigen_val_and_points(cfg.grf_modes, 2, sigma)
    Phi = RF._eigen_funcs(kl_points, inv.element_centroids)
    return sqrt_eigs[:, None] * Phi


def _run_adjoint(inv, scaled_Phi, y_obs, m_true, cfg, counters):
    """Discrete adjoint in the 128-dimensional KL coefficient space."""
    from scipy.optimize import minimize

    r = scaled_Phi.shape[0]

    def obj(xi):
        m = scaled_Phi.T @ xi
        J, grad_m, _, _ = inv.objective_and_gradient(m, y_obs)
        counters.forward(1)
        counters.adjoint(1)
        return J, scaled_Phi @ grad_m

    res = minimize(fun=obj, x0=np.zeros(r), jac=True, method="BFGS",
                   options=dict(maxiter=cfg.max_iter, gtol=cfg.gtol, disp=False,
                                method_bfgs=cfg.opt_method,
                                hess_inv0=np.eye(r), initial_scale=False))
    counters.n_fev = int(res.nfev)
    counters.n_iter = int(res.nit)
    counters.n_params = r
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(r)
    return scaled_Phi.T @ res.x


def _run_eki(inv, scaled_Phi, y_obs, cfg, counters, sigma_obs):
    """Ensemble Kalman inversion in the same KL space the adjoint uses."""
    from UKI import EKI

    r = scaled_Phi.shape[0]
    eki = EKI(J=cfg.eki_ensemble, param_dim=r, obs_dim=len(y_obs))

    def forward(xi):
        counters.forward(1)
        return inv.observe(inv.solve_forward(scaled_Phi.T @ xi))

    rng = np.random.RandomState(cfg.eki_seed)
    ensemble = rng.randn(cfg.eki_ensemble, r)
    R = (max(sigma_obs, 1e-12) ** 2) * np.eye(len(y_obs))
    for _ in range(cfg.eki_iters):
        ensemble = eki.analysis(ensemble, y_obs, forward, R)
    counters.n_iter = cfg.eki_iters
    counters.n_fev = counters.forward_solves
    counters.n_params = r
    return scaled_Phi.T @ ensemble.mean(axis=0)


_LAST_PINN = None   # set by _run_pinn so callers can read its history


def _run_pinn(inv, RF, coefs_true, obs_points, y_obs, cfg, counters):
    """PINN with a neural field for the unknown, H1 weight converted truth-free."""
    import torch
    from cfg import calibrate_C_mesh, pinn_reg_weights
    from darcy_pinn import DarcyPINN

    C_mesh, diag = calibrate_C_mesh(inv, RF, cfg, coefs_true=coefs_true)
    beta_pinn, gamma_pinn = pinn_reg_weights(cfg, C_mesh)
    print(f"C_mesh={C_mesh:.4f} ({diag.get('source')}) -> gamma_pinn={gamma_pinn:.4e}")

    def f_torch(x, y):
        return cfg.source_value * torch.ones_like(x)

    def m_true_func(x, y):
        pts = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
        return RF.sample(pts, coefs_true).flatten().reshape(np.asarray(x).shape)

    pinn = DarcyPINN(
        f_func=f_torch, data_points=obs_points, data_values=y_obs,
        beta=beta_pinn, gamma=gamma_pinn,
        hidden_dims=list(cfg.hidden_dims), head_dims_u=list(cfg.head_dims_u),
        head_dims_m=list(cfg.head_dims_m), device=cfg.device,
        m_true_func=m_true_func,
    )
    pinn.train_adam(n_iterations=cfg.n_adam_iters, lr=cfg.adam_lr,
                    resample_every=cfg.adam_resample_every,
                    n_pde=cfg.n_pde, n_bc=cfg.n_bc, n_reg=cfg.n_reg,
                    verbose=True, print_every=1000)
    pinn.train_bfgs(maxiter=cfg.bfgs_maxiter, n_epochs=cfg.n_bfgs_epochs,
                    n_restarts=cfg.n_bfgs_restarts, n_pde=cfg.n_pde,
                    n_bc=cfg.n_bc, n_reg=cfg.n_reg, method_bfgs=cfg.bfgs_method,
                    gtol=cfg.bfgs_gtol, ftol=cfg.bfgs_ftol,
                    target_loss=cfg.bfgs_target_loss,
                    adam_recovery_steps=cfg.adam_recovery_steps,
                    adam_recovery_lr=cfg.adam_recovery_lr, verbose=True)

    counters.n_params = sum(p.numel() for p in pinn.net.parameters())
    counters.hessian_bytes_analytic = ssbroyden_hessian_bytes(counters.n_params)
    # Real evaluation counts, taken from the PINN's own history rather than from
    # the nominal budget: the BFGS phase can restart or stop early, so the two
    # differ. R3.2 asks for the number actually performed.
    n_evals = len(pinn.history.get("loss", []))
    counters.n_fev = int(n_evals)
    counters.n_iter = int(pinn.history["iteration"][-1]) if n_evals else 0
    counters.residual(cfg.n_pde * max(n_evals, 1))
    global _LAST_PINN
    _LAST_PINN = pinn
    c = inv.element_centroids
    return pinn.predict_m(c[:, 0], c[:, 1]).ravel(), C_mesh, gamma_pinn


def run_one(row, index):
    require_l40s()

    from cfg import DarcyConfig
    import torch

    cfg = DarcyConfig(
        gamma=row["gamma"],
        noise_level=row["noise"],
        noise_seed=1000 + row["seed"],     # noise realization varies with the seed
        eki_seed=2000 + row["seed"],
        seed=row["seed"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    set_seed(cfg.seed)

    inv, RF, coefs_true, m_true, y_obs, u_clean, obs_points = _setup(cfg, row["seed"])
    scaled_Phi = _kl_basis(inv, RF, cfg)
    counters = Counters()
    sigma_obs = cfg.noise_level * np.max(np.abs(u_clean))

    C_mesh = gamma_pinn = None
    with RunTimer() as timer:
        if row["method"] == "adjoint":
            m_rec = _run_adjoint(inv, scaled_Phi, y_obs, m_true, cfg, counters)
            representation = "kl"
        elif row["method"] == "eki":
            m_rec = _run_eki(inv, scaled_Phi, y_obs, cfg, counters, sigma_obs)
            representation = "kl"
        else:
            m_rec, C_mesh, gamma_pinn = _run_pinn(
                inv, RF, coefs_true, obs_points, y_obs, cfg, counters)
            representation = "nn"

    # eps_u re-solves the FEM forward problem with the recovered field, for every
    # method, so the state error is measured the same way throughout.
    u_rec = inv.solve_forward(m_rec)
    eps_f = float(np.linalg.norm(m_rec - m_true) / np.linalg.norm(m_true))
    eps_u = float(np.linalg.norm(u_rec - u_clean) / np.linalg.norm(u_clean))

    return make_record(
        bundle="E", benchmark="darcy",
        method=row["method"], representation=representation,
        optimizer=cfg.opt_method if row["method"] != "eki" else "eki",
        arch=("kl%d" % cfg.grf_modes) if representation == "kl"
             else "x".join(str(h) for h in cfg.hidden_dims),
        seed=row["seed"], noise=row["noise"], gamma=row["gamma"], beta=cfg.beta,
        eps_f=eps_f, eps_u=eps_u, converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        sub=row["sub"], C_mesh=C_mesh, gamma_pinn=gamma_pinn,
        sigma_obs=float(sigma_obs), cfg_snapshot=cfg,
    )


if __name__ == "__main__":
    cli("E", rows, run_one)
