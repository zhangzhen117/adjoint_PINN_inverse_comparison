"""Figure 1 of the manuscript: Test 1 convergence histories.

Regenerates B_training_history.png from the bundle-A runs. Relative to the
submitted version there is one change, requested for referee comment R1.4: the
PINN forcing-error panel now carries **both** parameterizations, drawn with the
same solid/dashed convention already used in the adjoint panel beside it. The
representation x algorithm factorial can then be read straight off the convergence
histories -- the two grid curves plateau together, an order of magnitude above the
two neural-field curves, irrespective of which algorithm produced them.

Seed 0 is plotted so the figure remains a single-run convergence history, matching
the submitted version; the spread across seeds is reported in
Table~\\ref{tab:repr_factorial} rather than drawn here.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "A_runs")
TINY = 1.0e-30


def load(name):
    return np.load(os.path.join(RUNS, name), allow_pickle=True)


pinn_nn = load("pinn_pinn_nn_s0.npz")
pinn_gr = load("pinn_pinn_grid_s0.npz")
adj_nn = load("adj_adjoint_nn_s0.npz")
adj_gr = load("adj_adjoint_grid_s0.npz")


def iters(d, key):
    """Prefer the per-iteration trace; fall back to per-evaluation."""
    v = np.asarray(d[f"iter_{key}"], dtype=float)
    if v.size == 0:
        v = np.asarray(d[f"eval_{key}"], dtype=float)
    return v


t_pinn_nn, t_pinn_gr = float(pinn_nn["runtime_sec"]), float(pinn_gr["runtime_sec"])
t_adj_nn, t_adj_gr = float(adj_nn["runtime_sec"]), float(adj_gr["runtime_sec"])

loss_adj_nn, rf_adj_nn = iters(adj_nn, "loss"), iters(adj_nn, "rel_l2_f")
loss_adj_gr, rf_adj_gr = iters(adj_gr, "loss"), iters(adj_gr, "rel_l2_f")

ax_pinn_nn = np.arange(1, len(pinn_nn["bfgs_loss"]) + 1)
ax_pinn_gr = np.arange(1, len(pinn_gr["bfgs_rel_l2_f"]) + 1)
ax_adj_nn = np.arange(1, len(loss_adj_nn) + 1)
ax_adj_gr = np.arange(1, len(loss_adj_gr) + 1)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
axes = axes.ravel()

# -- panel 0: PINN loss components (neural-field parameterization) -------------
for key, lab in (("bfgs_loss", r"$\ell$"), ("bfgs_L_res", r"$\ell_{\mathrm{pde}}$"),
                 ("bfgs_L_ic", r"$\ell_{\mathrm{ic}}$"), ("bfgs_L_bc", r"$\ell_{\mathrm{bc}}$"),
                 ("bfgs_L_data", r"$\ell_{\mathrm{data}}$")):
    axes[0].semilogy(ax_pinn_nn, np.asarray(pinn_nn[key], float) + TINY, lw=2, label=lab)
axes[0].set_title(f"PINN losses, NN (runtime {t_pinn_nn:.0f} s)")
axes[0].set_ylabel("loss")
axes[0].legend()

# -- panel 1: adjoint objective, both parameterizations ------------------------
axes[1].semilogy(ax_adj_nn, loss_adj_nn + TINY, lw=2, label="NN")
axes[1].semilogy(ax_adj_gr, loss_adj_gr + TINY, "--", lw=2, label="Coarse mesh")
axes[1].set_title(f"Adjoint objectives (NN {t_adj_nn:.0f} s, Coarse mesh {t_adj_gr:.0f} s)")
axes[1].set_ylabel("loss")
axes[1].legend()

# -- panel 2: PINN forcing error, BOTH parameterizations (the R1.4 addition) ---
axes[2].semilogy(ax_pinn_nn, np.asarray(pinn_nn["bfgs_rel_l2_f"], float) + TINY,
                 lw=2, label="NN")
axes[2].semilogy(ax_pinn_gr, np.asarray(pinn_gr["bfgs_rel_l2_f"], float) + TINY,
                 "--", lw=2, label="Coarse mesh")
axes[2].set_title(r"Rel $L^2$ error of PINN forcing")
axes[2].set_ylabel(r"$\varepsilon_f$")
axes[2].legend()

# -- panel 3: adjoint forcing error, both parameterizations --------------------
axes[3].semilogy(ax_adj_nn, rf_adj_nn + TINY, lw=2, label="NN")
axes[3].semilogy(ax_adj_gr, rf_adj_gr + TINY, "--", lw=2, label="Coarse mesh")
axes[3].set_title(r"Rel $L^2$ error of adjoint forcing")
axes[3].set_ylabel(r"$\varepsilon_f$")
axes[3].legend()

# Shared y-range on the two error panels so the factorial is readable across them.
pos = np.concatenate([v[v > 0.0] for v in (
    np.asarray(pinn_nn["bfgs_rel_l2_f"], float), np.asarray(pinn_gr["bfgs_rel_l2_f"], float),
    rf_adj_nn, rf_adj_gr) if np.any(v > 0.0)])
lo = 10.0 ** np.floor(np.log10(pos.min()))
hi = 10.0 ** np.ceil(np.log10(pos.max()))
axes[2].set_ylim(lo, hi)
axes[3].set_ylim(lo, hi)

for ax in axes:
    ax.set_xlabel("Iteration")
    ax.grid(True, which="both", alpha=0.3)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                          numticks=100))
    ax.set_title(ax.get_title(), fontsize=13)
    ax.set_xlabel(ax.get_xlabel(), fontsize=13)
    ax.set_ylabel(ax.get_ylabel(), fontsize=13)
    ax.tick_params(axis="both", which="both", labelsize=13)
    lg = ax.get_legend()
    if lg is not None:
        for t in lg.get_texts():
            t.set_fontsize(13)

out_debug = os.path.join(REPO, "debug", "B_training_history.png")
os.makedirs(os.path.dirname(out_debug), exist_ok=True)
fig.savefig(out_debug, bbox_inches="tight", dpi=300)
print("wrote", out_debug)

print(f"\n{'series':>22} {'final eps_f':>12} {'runtime':>9}")
for lab, v, t in (("PINN / NN", pinn_nn["bfgs_rel_l2_f"], t_pinn_nn),
                  ("PINN / coarse mesh", pinn_gr["bfgs_rel_l2_f"], t_pinn_gr),
                  ("Adjoint / NN", rf_adj_nn, t_adj_nn),
                  ("Adjoint / coarse mesh", rf_adj_gr, t_adj_gr)):
    print(f"{lab:>22} {float(np.asarray(v)[-1]):>12.3e} {t:>8.0f}s")
