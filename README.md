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

## Layout

The tree follows the paper: four benchmark cases and the two ablation studies of
Appendix B.

```
cases/         the four benchmarks of Section 3
  case1_burgers/      case2_darcy/      case3_allencahn/      case4_cylinder/
ablations/     the two sensitivity studies of Appendix B
  b1_optimizer/                          b2_architecture/
sweeps/        the remaining referee sweeps, which feed the main-text tables
common/        seeding, instrumentation, the sweep runner, shared PINN pieces
analysis/      one script per manuscript figure
results/       one JSON line per run, uniform schema
slurm/         array drivers
```

## Benchmarks

| Test | Directory | Problem | Unknown recovered |
|------|-----------|---------|-------------------|
| 1 | `cases/case1_burgers/` | 1D viscous Burgers (periodic, Crank–Nicolson + Newton) | spatial forcing `f(x)` |
| 2 | `cases/case2_darcy/` | 2D Darcy flow, FEM (P2 elements) | log-permeability field `f(x)`, `k = exp(f)` |
| 3 | `cases/case3_allencahn/` | 3D Allen–Cahn (Neumann, spectral/FD) | state-dependent reaction force `f(u)` |
| 4 | `cases/case4_cylinder/` | 2D flow past a cylinder (Re = 100, vortex shedding) | viscosity `ν` (Reynolds number) |

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

Case 2 additionally includes Gaussian-random-field utilities (`GRF.py`) and an
Unscented Kalman Inversion baseline (`UKI.py`). Case 4 carries its OpenFOAM
mesh-refinement study in `cases/case4_cylinder/gridstudy/`, which establishes that
the Test 4 observations are grid-converged and holds the inversions run against
them; its raw case directories are gitignored, and `mkcase.py` with
`slurm/run_grid.slurm` regenerates them.

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
The scalar-control case (`cases/case4_cylinder/inverse_coarse.py` and
`cylinder_api.py`) deliberately uses plain `BFGS` instead, since the SSBroyden
update is singular for a one-dimensional control.

## Running

Each benchmark is self-contained. Run scripts from inside their directory, or open
the corresponding notebook:

```bash
cd cases/case1_burgers
jupyter notebook run_identification.ipynb
```

Case 4 also ships shell drivers (`run_cylinder.sh`, `run_pinn_inv_noise_sweep.sh`)
for batch runs.

## The two ablations

Appendix B reports two sensitivity studies, one per directory under `ablations/`.

| Study | Directory | Benchmark | Varies | Table | Referee |
|---|---|---|---|---|---|
| B.1 | `ablations/b1_optimizer/`    | Burgers    | Adam, SOAP, SSBroyden, each at its tuned learning rate | 6 | R3.7 |
| B.2 | `ablations/b2_architecture/` | Allen–Cahn | width, depth, activation (silu, L-LAAF), Fourier input encoding | 7 | R1.5, R3.4 |

`b1_optimizer` holds two drivers: `bundle_B` sweeps the learning rate on one seed
and `bundle_B1` reruns each optimizer at its tuned rate over five seeds, which is
the table. `b2_architecture` holds `bundle_F`.

## Supporting sweeps

The remaining referee sweeps feed the main-text tables rather than the appendix.
Every sweep, ablation or not, runs one row per (configuration x seed), dispatched
by `slurm/run_sweep.sh` as an array job, and writes one JSON line per run to
`results/<letter>.jsonl` under a uniform schema, so each table and figure in the
paper is generated from one dataframe.

| bundle | case | question | feeds | referee |
|---|---|---|---|---|
| A   | 1 Burgers    | representation x algorithm factorial               | Table 3 | R1.4       |
| C   | 4 cylinder   | does a modern PINN setup change the result?        | --      | R1.5, R3.4 |
| D   | 3, 4         | multi-initialization statistics                    | Table 2 | R1.7, R3.5 |
| D2  | 3 Allen-Cahn | seed statistics for the adjoint                    | Table 2 | R1.7, R3.5 |
| E   | 2 Darcy      | noise sensitivity and the gamma sweep              | Table 4 | R3.6       |
| E2x | 2 Darcy      | extend the gamma grid so the optimum is bracketed  | Table 4 | R3.6       |
| E2y | 2 Darcy      | complete the gamma sweep at the other noise levels | Table 4 | R3.6       |
| G   | 4 cylinder   | does converged reference data improve nu?          | Table 2 | --         |
| H   | 3 Allen-Cahn | PINN-warm-started adjoint restart, five seeds      | Sec. 3.3 | --        |

```bash
python -m sweeps.bundle_G --count               # number of rows
sbatch --array=0-9 slurm/run_sweep.sh G         # run them
sbatch --array=0-9 slurm/run_sweep.sh B1        # ablations dispatch the same way
```

`run_sweep.sh` resolves B and B1 to `ablations.b1_optimizer`, F to
`ablations.b2_architecture`, and everything else to `sweeps`, so the bundle letter
is all a caller needs.

## Figures

One script per manuscript figure, each writing directly into `paper_overleaf/`:

| script | figure |
|---|---|
| `analysis/plot_burgers_history.py`  | Fig. 1, `B_training_history.png` |
| `analysis/plot_darcy_history.py`    | Fig. 3, `D_training_history.png` |
| `analysis/plot_ac_history.py`       | Fig. 6, `AC_training_history.png` |
| `analysis/plot_cylinder_history.py` | Fig. 9, `C_optimization_history_4panel.png` |

The remaining `analysis/plot_*.py` are diagnostics that write to `debug/` and are
not manuscript figures.

## Repository contents

Saved optimization histories (`history/*.npz`, `*.pt`) and generated figures
(`figures/*.png`) are committed so the results are reproducible without re-running
the (sometimes long) optimizations.
