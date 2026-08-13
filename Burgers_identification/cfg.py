import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class BurgersConfig:
    # ---------------- Grid / physics ----------------
    L: float = 2 * np.pi
    t_final: float = 1.0
    N: int = 512
    dt: float = 1e-2
    # Relative errors from convergence test: relL2=4.19452e-05, relLinf=1.01438e-04
    nu: float = 1.5e-2

    # ---------------- Regularization ----------------
    alpha_reg: float = 0.0

    # ---------------- Run identity ----------------
    # Seeds network initialization and collocation sampling. Previously the
    # notebook never seeded torch at all, so runs were not reproducible and no
    # multi-initialization statistics were possible (referees R1.7, R3.5).
    seed: int = 0

    # ---------------- Representation of the unknown f(x) ----------------
    # "nn"   -- f is an MLP, N_f = 1153 weights
    # "grid" -- f is n_coarse nodal values on a uniform periodic mesh, linearly
    #           interpolated to the forward grid
    # Both the adjoint and the PINN can use either, which is what makes the
    # representation x algorithm factorial of bundle A possible.
    representation: str = "nn"
    n_coarse: int = 64

    # ---------------- Optimizer selection (bundle B, referee R3.7) ------------
    # Adjoint side: "ssbroyden2" | "bfgs" | "lbfgsb". SSBroyden carries a dense
    # N_f x N_f inverse Hessian, so it is the one that does not scale; the other two
    # are the fallbacks whose ranking the ablation checks.
    adjoint_optimizer: str = "ssbroyden2"
    # PINN side: "ssbroyden2" | "adam" | "soap".
    pinn_optimizer: str = "ssbroyden2"

    # Learning rates. These must be tunable per optimizer: SOAP at lr=1e-3 is ~14x
    # worse than Adam on a smooth fit, while at its default 3e-3 it is ~120x better,
    # so a single shared learning rate would measure the choice of lr rather than
    # the choice of optimizer.
    pinn_adam_lr: float = 1e-3      # was hard-coded as a literal in train_pinn
    pinn_soap_lr: float = 3e-3      # SOAP's published default

    # ---- run-to-saturation protocol for the first-order optimizers ----
    # Adam and SOAP are run until the loss genuinely saturates, however long that
    # takes, rather than to a fixed budget: a budget comparison answers "who is
    # ahead at time T", which is not what R3.7 asks. The wall-clock below is only a
    # safety ceiling so a pathological run cannot hold a node indefinitely.
    walltime_cap_s: float | None = 8 * 3600.0

    # ReduceLROnPlateau schedule. Decaying the rate is what lets a first-order
    # method actually settle instead of oscillating around the minimum forever.
    pinn_lr_factor: float = 0.3
    pinn_lr_patience: int = 5_000      # steps without improvement before decaying
    pinn_lr_min_ratio: float = 1e-5    # stop once lr falls below lr0 * this
    pinn_plateau_steps: int = 50_000   # steps without improvement before stopping

    # Diagnostics (relative errors) are expensive: computing them every step made
    # them a significant share of a long Adam run's wall-clock, which would inflate
    # its cost relative to the far fewer evaluations of the quasi-Newton path. They
    # are recorded on a stride instead, and the timer excludes them.
    pinn_diag_every: int = 100

    # Adjoint-side plateau stop for the optimizer ablation. 0 disables it and
    # the run goes to scipy_adj_nn_maxiter, which is the paper's behaviour.
    # Every optimizer is still descending at iteration 400, so a fixed budget
    # would compare them mid-flight rather than at convergence.
    adj_plateau_window: int = 0
    adj_plateau_rtol: float = 1e-8

    # ---------------- Adjoint optimizer (SciPy) ----------------
    scipy_adj_nn_maxiter: int = 400
    scipy_ftol: float = 0
    scipy_gtol: float = 0
    scipy_method_bfgs: str = "SSBroyden2"
    scipy_verbose: bool = False
    scipy_disp: bool = False
    # NOTE: scipy_target_loss and scipy_bfgs_restarts were never referenced by any
    # code path; they are removed so they cannot be mistaken for live settings.
    # The adjoint runs to scipy_adj_nn_maxiter with ftol = gtol = 0.

    GD_warmup_lr: float = 1e-3

    # ---------------- PINN optimizer (SciPy) ----------------
    scipy_pinn_adam_warmup_steps: int = 1000
    scipy_pinn_maxiter: int = 200
    scipy_pinn_epochs: int = 20

    # ---------------- Device ----------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Paths ----------------
    adj_nn_path: str = "history/adj_nn.npz"
    adj_grid_path: str = "history/adj_grid.npz"
    pinn_scipy_path: str = "history/pinn_scipy.npz"
    pinn_scipy_model: str = "history/pinn_scipy.pt"

    # ---------------- Loss weights ----------------
    w_res: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    w_dataT: float = 1.0

    # ---------------- Sampling sizes ----------------
    n_interior: int = 20000
    n_bc_t: int = 512

    # ---------------- NN model sizes ----------------
    hidden_U: int = 32
    layers_U: int = 2
    act_U: str = "tanh"

    hidden_S: int = 32
    layers_S: int = 2
    act_S: str = "tanh"
