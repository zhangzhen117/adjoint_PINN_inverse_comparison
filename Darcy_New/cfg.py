"""Configuration for the 2D Darcy benchmark (Test 2).

Darcy was the only benchmark whose parameters lived as literals in a notebook cell
rather than in a dataclass, which is what made it impossible to sweep. Every value
here was previously hard-coded in ``darcy_noise.ipynb`` cells 7, 9, 14 and 23; the
notebook now imports this object so that the notebook run and the swept runs are
guaranteed to use identical settings.

Two behavioural changes are recorded explicitly rather than made silently:

1. ``max_iter`` is now honoured. The notebook set ``max_iter = 500`` in its control
   cell and then passed a hard-coded ``maxiter=50`` to ``sp_minimize``, with the 500
   commented out on the line above -- so the adjoint actually ran 50 iterations while
   the control cell advertised 500. The default below is the value that was really
   used.

2. ``C_mesh_source`` defaults to ``"prior"``. See :func:`calibrate_C_mesh`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np


@dataclass
class DarcyConfig:
    # ---------------- Mesh / FEM ----------------
    nx: int = 32
    ny: int = 32
    element_degree: int = 2          # P2; gives N_x = 4225 DOFs, 2048 elements
    linear_solver: str = "direct"
    # Constant volumetric source g = 100, matching the adjoint, PINN and EKI paths.
    source_value: float = 100.0

    # ---------------- Gaussian random field (prior and truth) ----------------
    grf_tau: float = 3.0
    grf_alpha: float = 2.0
    grf_sigma: float | None = None   # None -> RF derives it from tau/alpha
    grf_modes: int = 128             # KL truncation r
    truth_seed: int = 1234           # seed for the coefficients defining f*

    # ---------------- Observations ----------------
    n_obs_x: int = 9
    n_obs_y: int = 9
    obs_margin: float = 0.1
    noise_level: float = 0.01        # 1% of max|u|
    noise_type: str = "max"
    noise_seed: int = 42

    # ---------------- Regularization ----------------
    beta: float = 0.0                # Tikhonov, penalizes ||m||^2
    gamma: float = 1.0e-3            # graph-H1 seminorm, penalizes ||grad m||^2

    # How the graph-H1 -> continuous-H1 conversion constant is obtained when
    # translating `gamma` into the PINN's `gamma_pinn`. See calibrate_C_mesh.
    #   "prior" -- average over fresh GRF draws (default; uses no ground truth)
    #   "truth" -- the original notebook behaviour, retained for auditing only
    #   "fixed" -- take C_mesh_value verbatim
    C_mesh_source: Literal["prior", "truth", "fixed"] = "prior"
    C_mesh_samples: int = 8
    C_mesh_seed: int = 20260808      # deliberately != truth_seed
    C_mesh_value: float | None = None

    # ---------------- Adjoint optimizer ----------------
    max_iter: int = 50               # the value the notebook actually used
    opt_method: str = "SSBroyden2"
    gtol: float = 0.0

    # ---------------- EKI baseline ----------------
    eki_ensemble: int = 2000
    eki_iters: int = 5
    eki_seed: int = 42

    # ---------------- PINN ----------------
    hidden_dims: tuple[int, ...] = (32, 32, 32)
    head_dims_u: tuple[int, ...] = ()
    head_dims_m: tuple[int, ...] = (32,)
    n_adam_iters: int = 2000
    adam_lr: float = 1e-3
    adam_resample_every: int = 200
    n_bfgs_epochs: int = 10
    n_bfgs_restarts: int = 1
    bfgs_maxiter: int = 200
    bfgs_method: str = "SSBroyden2"
    bfgs_gtol: float = 0.0
    bfgs_ftol: float = 0.0
    bfgs_target_loss: float = 10.0
    adam_recovery_steps: int = 20
    adam_recovery_lr: float = 1e-4
    n_pde: int = 3000
    n_bc: int = 200
    n_reg: int = 1000
    # Representation of the unknown used by the PINN: a neural field, or the same
    # 128-mode KL basis the adjoint and EKI use (for the representation ablation).
    pinn_representation: Literal["nn", "kl"] = "nn"

    # ---------------- Run identity ----------------
    seed: int = 0                    # network init / sampling seed
    device: str = "cuda"

    # ---------------- Paths ----------------
    hist_dir: str = "history"
    fig_dir: str = "figures"

    @property
    def n_obs(self) -> int:
        return self.n_obs_x * self.n_obs_y

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate_C_mesh(
    inv,
    RF,
    cfg: DarcyConfig,
    m_true: np.ndarray | None = None,
    coefs_true: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Conversion constant between the adjoint's graph-H1 seminorm and the PINN's.

    The adjoint penalizes a *sum* over mesh edges,
    ``sum_edges (m_i - m_j)^2 / h_ij``, while the PINN penalizes a *mean* over
    collocation points, ``mean(|grad m|^2)``. Matching the two therefore needs the
    ratio ``C_mesh = graph_H1(m) / int|grad m|^2``, after which
    ``gamma_pinn = 2 * gamma * C_mesh / N_obs``.

    ``C_mesh`` is a property of the *mesh* (it scales like 1/h), not of any
    particular field, so any field of the right smoothness class estimates it. The
    original notebook estimated it on ``m_true`` -- which means the ground truth was
    setting a hyperparameter of one of the methods being compared. That is a fair
    thing for a referee to object to, so the default here draws several independent
    samples from the *prior* instead and averages. No ground truth is touched.

    ``C_mesh_source="truth"`` reproduces the original behaviour so the two can be
    compared and the change audited; the returned diagnostics carry both.

    Returns
    -------
    (C_mesh, diagnostics)
    """
    if cfg.C_mesh_source == "fixed":
        if cfg.C_mesh_value is None:
            raise ValueError('C_mesh_source="fixed" requires C_mesh_value to be set')
        return float(cfg.C_mesh_value), {"source": "fixed"}

    kappa_mesh = inv.element_centroids

    def ratio_for(coefs: np.ndarray) -> tuple[float, float, float]:
        m_elem = RF.sample(kappa_mesh, coefs).flatten()
        graph_h1 = sum(
            (m_elem[e_i] - m_elem[e_j]) ** 2 / h_ij
            for e_i, e_j, h_ij in inv.element_neighbors
        )
        # Continuous int|grad m|^2 over the unit square, by fine-grid differences.
        ng = 200
        grid = np.linspace(0.0, 1.0, ng)
        X, Y = np.meshgrid(grid, grid)
        m_smooth = RF.sample(
            np.column_stack([X.ravel(), Y.ravel()]), coefs
        ).reshape(ng, ng)
        dm_dy, dm_dx = np.gradient(m_smooth, grid, grid)
        int_grad_sq = np.mean(dm_dx**2 + dm_dy**2)
        return graph_h1 / int_grad_sq, graph_h1, int_grad_sq

    if cfg.C_mesh_source == "truth":
        if coefs_true is None:
            raise ValueError('C_mesh_source="truth" requires coefs_true')
        C, g, i = ratio_for(coefs_true)
        return float(C), {"source": "truth", "graph_h1": g, "int_grad_sq": i, "n_samples": 1}

    # Default: estimate from prior draws, disjoint from the truth seed.
    rng = np.random.default_rng(cfg.C_mesh_seed)
    ratios = [ratio_for(rng.standard_normal(cfg.grf_modes))[0] for _ in range(cfg.C_mesh_samples)]
    ratios = np.asarray(ratios, dtype=float)

    diagnostics = {
        "source": "prior",
        "n_samples": int(cfg.C_mesh_samples),
        "C_mesh_mean": float(ratios.mean()),
        "C_mesh_std": float(ratios.std(ddof=1)) if ratios.size > 1 else 0.0,
        "C_mesh_rel_spread": float(ratios.std(ddof=1) / ratios.mean()) if ratios.size > 1 else 0.0,
    }
    # Record the truth-based value alongside, when available, so the switch away
    # from it is auditable rather than merely asserted.
    if coefs_true is not None:
        diagnostics["C_mesh_truth"] = float(ratio_for(coefs_true)[0])

    return float(ratios.mean()), diagnostics


def pinn_reg_weights(cfg: DarcyConfig, C_mesh: float) -> tuple[float, float]:
    """Translate the adjoint's (beta, gamma) into the PINN's (beta_pinn, gamma_pinn)."""
    beta_pinn = cfg.beta
    gamma_pinn = 2.0 * cfg.gamma * C_mesh / cfg.n_obs
    return beta_pinn, gamma_pinn
