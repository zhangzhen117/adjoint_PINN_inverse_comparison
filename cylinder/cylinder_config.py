"""Single source of truth for the cylinder viscosity-identification study.

Every control flag that used to live (hardcoded) at module scope across
grad_check_probes.py / make_probe_obs2.py / legacy inverse scripts / the PINN scripts
is collected here in one dataclass, mirroring Burgers_1D/BurgersConfig.py. The
notebook run_cylinder.ipynb and cylinder_api.CylinderAPI read everything from a
single CylinderRunConfig instance.

Baseline physics: Re = 100  (nu_true = U*D/Re = 1/100 = 0.01), vortex-shedding
regime. The forward solver itself is left untouched (cylinder_solver.py); the
helper solver_cfg() hands it a plain cylinder_solver.CylinderConfig.
"""
from dataclasses import dataclass, field, asdict
import numpy as np

import cylinder_solver


@dataclass
class CylinderRunConfig:
    # ----------------------------------------------------------------- physics
    Re: float = 100.0                 # <-- baseline Reynolds number (was 60)
    U_inf: float = 1.0
    D: float = 1.0

    # --------------------------------------------------------------- geometry
    x0: float = -3.0; x1: float = 10.0
    y0: float = -3.0; y1: float = 3.0

    # ----------------------------------------------------- mesh refinement lvls
    # coarse mesh -> optional fast adjoint experiments
    h_cyl_coarse: float = 0.06; h_wake_coarse: float = 0.12; h_far_coarse: float = 0.6
    # production mesh -> default forward / observation / adjoint / PINN reference mesh
    h_cyl_prod: float = 0.04;  h_wake_prod: float = 0.08;  h_far_prod: float = 0.5
    eps_p: float = 1e-10

    # ------------------------------------------------------------------- time
    dt: float = 0.005                  # fixed step for obs / adjoint window

    # ------------------------------------------- observation layout (SHARED!)
    # number AND position of probes live here; every stage that builds or
    # consumes observations (gradient check, inversion, obs builder, inverse
    # PINN) calls cfg.probe_xy() / cfg.obs_steps() so the probe set is identical.
    probe_x_lo: float = 1; probe_x_hi: float = 3; n_px: int = 4
    probe_y_lo: float = -1; probe_y_hi: float = 1; n_py: int = 4
    obs_stride: int = 100             # steps between observed snapshots
    n_obs: int = 10                   # number of observed snapshots
    obs_noise: float = 0.00            # additive noise std on observations
    obs_seed: int = 42

    # --------------------------------------------------- warmup / IC generation
    # forward warmup to reach the saturated shedding limit cycle; its final
    # state becomes the (fixed, known) inverse-problem IC saved to saturated.npz.
    warm_T: float = 80.0              # convective times to saturate (Re=100)
    warm_save_dt: float = 5.0         # snapshot spacing during saturation warmup
    ramp_T: float = 2.0              # smooth inlet ramp 0 -> U_inf
    blob_t: float = 8.0              # time to inject the symmetry-breaking blob
    blob_amp: float = 0.1
    cfl_target: float = 0.9          # adaptive-dt target during warmup
    ref_save_dt: float = 0.1         # snapshot spacing for the period reference

    # ----------------------------------------------- gradient-check (adjoint)
    gc_eps: float = 1e-6                              # central-FD step in theta
    gc_fracs: tuple = (0.7, 0.85, 1.2, 1.5)          # nu/nu_true offsets tested

    # --------------------------------------------------- adjoint inversion
    adj_mesh: str = "prod"           # mesh level for gradient check + inversion
                                     #   'prod' by default, 'coarse' only for fast tests
    inv_nu0_factor: float = 100      # start at nu0 = factor * nu_true (Re~100)
    inv_maxiter: int = 10
    inv_gtol: float = 1e-5
    inv_method_bfgs: str = "BFGS"    # SSBroyden* is singular for a scalar control

    # ----------------------------------------------------------- PINN (shared)
    hidden: int = 32                 # <-- 32 x 3 for BOTH forward and inverse
    layers: int = 3
    margin: float = 0.02             # clearance for PDE collocation (no surface)
    near_frac: float = 0.0           # <-- no near-cylinder oversampling anywhere
    r_near: float = 0.75
    n_pde: int = 10000
    n_bc: int = 2000
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 10.0
    w_data: float = 10.0             # inverse only (probe-data misfit)
    adam_epochs: int = 1000          # forward Adam epochs
    adam_blocks: int = 2             # inverse Adam blocks ...
    adam_per: int = 500              # ... x steps per block (= 1000 steps)
    adam_lr: float = 1e-3
    adam_rec: int = 200              # forward Adam record period
    print_every: int = 20            # SSBroyden record/print period
    n_restarts_fwd: int = 20
    n_restarts_inv: int = 10
    maxiter: int = 200               # SSBroyden max iters per restart
    # Plateau stop across SSBroyden restarts; 0 keeps the published fixed budget.
    inv_restart_plateau: int = 0
    inv_restart_rtol: float = 1e-6
    gtol_fwd: float = 1e-10
    gtol_inv: float = 1e-12
    nu0_pinn: float =  1.0          # inverse PINN cold-start nu (below truth)
    seed_fwd: int = 45
    seed_inv: int = 44

    # ---- input encoding (bundle C2, referee R3.4) ----
    # R3.4 asks whether the conclusions depend on architecture, naming Fourier
    # features and adaptive activations specifically. "none" is the plain tanh MLP
    # used throughout the paper; "fourier" prepends a random Fourier feature map,
    # sin/cos(2*pi*B*x) with B ~ N(0, fourier_scale^2), the standard remedy for the
    # spectral bias of coordinate networks. Note that on this benchmark the unknown
    # is the scalar nu, so the encoding affects the *state* network only.
    pinn_encoding: str = "none"          # "none" | "fourier"
    fourier_features: int = 32           # yields 2*fourier_features channels
    fourier_scale: float = 1.0

    # ---- alternative training setup, after jaxpi2 (bundle C, second arm) ----
    # github.com/sifanexisted/jaxpi2: gated "modified" MLP + SOAP + gradient-norm
    # loss balancing + pseudo-time relaxation of the residual. Under the paper's
    # own setup ("paper") only the production 32x3 network recovers nu at all, so
    # running the same width/depth grid under a stronger setup is what makes the
    # architecture question answerable rather than a measure of training fragility.
    #
    # Sized to the paper, NOT to jaxpi2's bfs_flow example: ModifiedMlp at 32x3
    # without Fourier is 2563 parameters against the paper's 2307, whereas the
    # reference 3x256+Fourier(256) config would be 526k.
    # Three cases are compared in bundle C:
    #   "paper"        -- the published setup: plain tanh MLP, Adam -> SSBroyden
    #   "modern"       -- all of the jaxpi2 techniques at once: gated ModifiedMlp
    #                     + SOAP + gradient-norm balancing + pseudo-time stepping
    #   "vanilla_adam" -- plain tanh MLP trained with Adam alone, to show that a
    #                     vanilla PINN does not solve the cylinder problem
    pinn_setup: str = "paper"      # "paper" | "modern" | "vanilla_adam"
    jaxpi_arch: str = "modified"   # "modified" | "plain"
    jaxpi_optimizer: str = "soap"  # "soap" | "adam"
    jaxpi_activation: str = "swish"
    jaxpi_fourier: bool = False          # embed_dim = hidden when enabled
    jaxpi_steps: int = 60_000
    jaxpi_lr: float = 3e-3            # cyl_pinn_pt.py default
    # 0.9 over 60k steps is only an 8x total decay, which left nu oscillating with
    # ~1e-2 amplitude at the end of training -- the run passed through eps_nu ~ 3e-5
    # but never settled there, and no truth-free criterion could pick that iterate.
    # 0.7 gives ~1000x total decay in the same step budget, so the iterate freezes.
    jaxpi_decay_rate: float = 0.7
    jaxpi_decay_steps: int = 3000
    jaxpi_warmup_steps: int = 2000
    jaxpi_resample_every: int = 1000

    # gradient-norm loss balancing. NOTE: the manuscript states all loss weights
    # are unity; this is adaptive balancing and bears directly on referee R1.6.
    jaxpi_gradnorm: bool = True
    jaxpi_gradnorm_every: int = 500
    jaxpi_gradnorm_momentum: float = 0.9

    # pseudo-time relaxation of the PDE residual
    jaxpi_pseudotime: bool = True
    # Matched to cyl_pinn_pt.py: theta recomputed every 200 steps with momentum
    # 0.7, while the previous iterate itself is refreshed every step.
    jaxpi_pts_every: int = 200
    jaxpi_pts_momentum: float = 0.7

    # ----------------------------------------------------------------- device
    device: str = field(default_factory=lambda: _torch_device())

    # ------------------------------------------------------------------- paths
    hist_dir: str = "history"
    fig_dir: str = "figures"
    saturated_path: str = "history/saturated.npz"     # regenerated IC + period ref
    obs_path: str = "history/probe_obs.npz"
    grad_check_path: str = "history/grad_check.npz"
    inverse_path: str = "history/inverse_adjoint.npz"
    inverse_restart_path: str = "history/inverse_adjoint_restart.npz"
    forward_path: str = "history/forward.npz"
    pinn_fwd_npz: str = "history/pinn_fwd.npz"
    pinn_fwd_model: str = "history/pinn_fwd_model.pt"
    pinn_inv_npz: str = "history/pinn_inv.npz"
    pinn_inv_model: str = "history/pinn_inv_model.pt"

    # =====================================================================
    # derived quantities
    # =====================================================================
    @property
    def nu_true(self) -> float:
        return self.U_inf * self.D / self.Re

    @property
    def R(self) -> float:
        return 0.5 * self.D

    @property
    def K(self) -> int:
        """Number of fixed-dt steps spanning the observation window."""
        return self.obs_stride * self.n_obs

    @property
    def T_obs(self) -> float:
        return self.K * self.dt

    def obs_steps(self):
        """Step indices at which probes are observed: stride, 2*stride, ..., K."""
        return list(range(self.obs_stride, self.K + 1, self.obs_stride))

    def probe_xy(self):
        """(n_px*n_py, 2) probe coordinates. Order = [(x, y) for x in PX for y
        in PY] -- matches build_probe_operator / make_probe_obs2 column order."""
        PX = np.linspace(self.probe_x_lo, self.probe_x_hi, self.n_px)
        PY = np.linspace(self.probe_y_lo, self.probe_y_hi, self.n_py)
        return np.array([(x, y) for x in PX for y in PY])

    def solver_cfg(self, level: str = "prod"):
        """Return a cylinder_solver.CylinderConfig for create_solver(). level is
        'coarse' (optional fast adjoint runs) or 'prod' (default forward / obs /
        adjoint / PINN runs)."""
        if level == "coarse":
            hc, hw, hf = self.h_cyl_coarse, self.h_wake_coarse, self.h_far_coarse
        elif level == "prod":
            hc, hw, hf = self.h_cyl_prod, self.h_wake_prod, self.h_far_prod
        else:
            raise ValueError("level must be 'coarse' or 'prod'")
        return cylinder_solver.CylinderConfig(
            Re=self.Re, U_inf=self.U_inf, D=self.D,
            x0=self.x0, x1=self.x1, y0=self.y0, y1=self.y1,
            h_cyl=hc, h_wake=hw, h_far=hf, dt=self.dt, eps_p=self.eps_p)

    def snapshot(self):
        """Plain-dict copy for embedding in saved npz files (cfg_snapshot)."""
        return asdict(self)

    # ------------------------------------------------------- quick-mode helper
    def quick(self):
        """Return a shallow copy shrunk for a fast smoke run (mirrors the
        Burgers notebook using pinn_adam_epochs=10). Heavy stages stay correct
        but cheap so the whole notebook executes end-to-end in minutes."""
        import copy
        c = copy.deepcopy(self)
        c.warm_T = 15.0
        c.n_obs = 4; c.obs_stride = 30           # K = 120, T_obs = 1.2
        c.inv_maxiter = 10
        c.adam_epochs = 200
        c.adam_blocks = 1; c.adam_per = 200
        c.n_pde = 4000; c.n_bc = 200
        c.n_restarts_fwd = 2; c.n_restarts_inv = 2
        c.maxiter = 50
        return c


def _torch_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
