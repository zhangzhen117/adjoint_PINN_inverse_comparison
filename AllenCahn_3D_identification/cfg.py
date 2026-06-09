import torch
from dataclasses import dataclass


@dataclass
class AllenCahn3DConfig:
    # Domain and grid
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    t_final: float = 3.0 
    Nx: int = 129
    Ny: int = 129
    Nz: int = 129
    dt: float = 5e-2
    eps: float = 1.0e-2

    # Regularization / optimization
    alpha_reg: float = 0.0
    scipy_adj_nn_maxiter: int = 200
    scipy_adj_nn_maxfun: int = 500
    scipy_pinn_adam_warmup_steps: int = 500
    scipy_pinn_maxiter: int = 200
    scipy_pinn_epochs: int = 15
    scipy_ftol: float = 0.0
    scipy_gtol: float = 0.0
    scipy_method_bfgs: str = "SSBroyden2"
    scipy_verbose: bool = False
    scipy_disp: bool = False
    GD_warmup_steps: int = 200
    GD_warmup_lr: float = 1.0e-3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    is_load: bool = False

    # Loss weights
    w_res: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    w_dataT: float = 1.0

    # Sampling sizes for PINN
    n_interior: int = 20000
    n_bc_t: int = 5000
    n_data_t: int = 80000

    # Network sizes
    hidden_U: int = 32
    layers_U: int = 3
    act_U: str = "tanh"
    hidden_S: int = 32
    layers_S: int = 2
    act_S: str = "tanh"

    # Output paths
    adj_nn_path: str = "history/ac3d_adj_nn.npz"
    adj_nn_model: str = "history/ac3d_adj_nn.pt"
    pinn_scipy_path: str = "history/ac3d_pinn_scipy.npz"
    pinn_scipy_model: str = "history/ac3d_pinn_scipy.pt"
