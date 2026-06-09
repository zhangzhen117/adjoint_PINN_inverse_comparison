# Adjoint vs. PINN for PDE Inverse Problems

A comparison study of two approaches to PDE-constrained inverse problems:

1. **Discrete-adjoint optimization** — a neural network parameterizes the unknown
   field; gradients of the data-misfit objective are computed by back-propagating
   through the (differentiable) forward solver, and the network is trained with a
   second-order optimizer (`scipy.optimize.minimize`, L-BFGS-B / trust-region).
2. **Physics-Informed Neural Networks (PINNs)** — the solution field and the unknown
   field are learned jointly by minimizing a combined data + PDE-residual loss,
   again driven by `scipy.optimize.minimize` over the torch parameters.

Both methods recover an unknown source / coefficient field `f` from sparse or noisy
observations, and the repository compares their accuracy and convergence on four
PDE benchmarks.

## Benchmarks

| Directory | Problem | Unknown recovered |
|-----------|---------|-------------------|
| `Burgers_identification/` | 1D viscous Burgers (periodic, Crank–Nicolson + Newton) | spatial forcing `f(x)` |
| `AllenCahn_3D_identification/` | 3D Allen–Cahn (Neumann, spectral/FD) | state-dependent reaction force `f(u)` |
| `Darcy_New/` | 2D Darcy flow, FEM (P2 elements) | log-permeability field `f(x)`, `k = exp(f)` |
| `cylinder/` | 2D flow past a cylinder (Re = 100, vortex shedding) | viscosity `ν` (Reynolds number) |

In every case except the cylinder, the unknown `f` is an infinite-dimensional
field, not a scalar parameter: a function of space in Burgers (`f(x)`) and Darcy
(`f(x)`, with permeability `k = exp(f)`), and a function of the state itself in
Allen–Cahn (`f(u)`). Only the cylinder case recovers a single scalar (`ν`).

Each benchmark directory follows the same layout:

- `solver.py` / `*_solver.py` — forward PDE solver (the differentiated model).
- `adjoint_operator.py` / `*_adjoint.py` — discrete-adjoint inverse solver.
- `PINN.py` / `*_pinn.py` — PINN inverse solver.
- `cfg.py` / `*_config.py` — single dataclass holding all run parameters.
- `run_*.ipynb` — notebook that runs both methods and produces the figures.
- `figures/` — generated plots; `history/` — saved optimization traces and models.

The `Darcy_New/` case additionally includes Gaussian-random-field utilities
(`GRF.py`) and an Unscented Kalman Inversion baseline (`UKI.py`).

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+. The code sets `torch.set_default_dtype(torch.float64)`
throughout, so a CPU-only torch build is sufficient.

### Required: patched SciPy for the SSBroyden optimizer

Every benchmark drives the optimizer with the **self-scaled Broyden** update,
invoked as:

```python
minimize(fun, x0, jac=True, method="BFGS", options={"method_bfgs": "SSBroyden2"})
```

The `method_bfgs` / `SSBroyden2` option is **not part of stock SciPy**. It is
provided by the patched `_optimize.py` from
[EliKiani/Optimizing_the_Optimizer_PINNs](https://github.com/EliKiani/Optimizing_the_Optimizer_PINNs)
(Self-scaled Quasi-Newton methods for PINNs). To run this code you must replace
the `_optimize.py` in your SciPy install with that file:

```bash
# locate your installed scipy
python -c "import scipy.optimize as o; print(o._optimize.__file__)"
# back up the original, then overwrite it with the patched _optimize.py
```

Without the patch, `scipy.optimize.minimize` ignores the `method_bfgs` option and
silently falls back to ordinary BFGS, so results will not match the paper.
The scalar-control cases (`cylinder/inverse_coarse.py`, `cylinder/cylinder_api.py`)
deliberately use plain `BFGS` instead, since the SSBroyden update is singular for a
one-dimensional control.

## Running

Each benchmark is self-contained. Run scripts from inside their directory, or open
the corresponding notebook:

```bash
cd Burgers_identification
jupyter notebook run_identification.ipynb
```

The `cylinder/` case also ships shell drivers (`run_cylinder.sh`,
`run_pinn_inv_noise_sweep.sh`) for batch runs.

## Repository contents

Saved optimization histories (`history/*.npz`, `*.pt`) and generated figures
(`figures/*.png`) are committed so the results are reproducible without re-running
the (sometimes long) optimizations.
