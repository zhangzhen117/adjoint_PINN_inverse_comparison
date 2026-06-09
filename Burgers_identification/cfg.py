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

    # ---------------- Adjoint optimizer (SciPy) ----------------
    scipy_adj_nn_maxiter: int = 400
    scipy_ftol: float = 0
    scipy_gtol: float = 0
    scipy_method_bfgs: str = "SSBroyden2"
    scipy_verbose: bool = False
    scipy_disp: bool = False
    scipy_target_loss: float = 1e-6
    scipy_bfgs_restarts: int = 10

    GD_warmup_lr: float = 1e-3

    # ---------------- PINN optimizer (SciPy) ----------------
    scipy_pinn_adam_warmup_steps: int = 1000
    scipy_pinn_maxiter: int = 200
    scipy_pinn_epochs: int = 20

    # ---------------- Device ----------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Paths ----------------
    adj_nn_path: str = "history/adj_nn.npz"
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
