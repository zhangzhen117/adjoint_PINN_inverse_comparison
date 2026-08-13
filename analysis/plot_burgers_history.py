"""Test 1 convergence histories over five seeds, for Figure 1.

Same four panels and the same solid/dashed convention as the current figure --
PINN losses, adjoint objectives, and the two forcing-error panels each carrying
both parameterizations -- but drawn from all five initializations rather than from
seed 0 alone.

The representation x algorithm factorial reads off the two error panels: the
coarse-grid curves (dashed) plateau together an order of magnitude above the
neural-field curves (solid), irrespective of which algorithm produced them.
Showing the seed bands makes the second half of that claim stronger than a single
run could, because the grid curves land on top of each other while the adjoint's
neural-field band is wide enough to overlap the PINN's.

The adjoint on the coarse grid starts from s_coarse = 0 and has no random element,
so it is a single deterministic run; every other arm is five seeds. That asymmetry
is in the panel labels, not left to the reader.
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "A_runs")
# Written straight into the manuscript checkout: one script per figure,
# no "_new" copy to diverge from what the paper actually shows.
OUT = os.path.join(REPO, "paper_overleaf", "B_training_history.png")
TINY = 1.0e-30


def series(pattern, key, evalkey=None):
    """(n_seeds, n) truncated to the shortest run, plus the runtimes."""
    files = sorted(glob.glob(os.path.join(RUNS, pattern)))
    assert files, f"no runs matching {pattern}"
    ys, ts = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if key in d.files:
            v = np.asarray(d[key], float)
        else:                                   # adjoint: per-iteration else per-eval
            v = np.asarray(d[f"iter_{evalkey}"], float)
            if v.size == 0:
                v = np.asarray(d[f"eval_{evalkey}"], float)
        ys.append(v)
        ts.append(float(d["runtime_sec"]))
    n = min(y.size for y in ys)
    return np.stack([y[:n] for y in ys]) + TINY, np.array(ts), n


def band(ax, Y, col, ls="-", label=None):
    x = np.arange(1, Y.shape[1] + 1)
    if Y.shape[0] > 1:
        for y in Y:
            ax.semilogy(x, y, color=col, ls=ls, lw=0.7, alpha=0.28, zorder=2)
    ax.semilogy(x, np.median(Y, axis=0), color=col, ls=ls, lw=2.2, label=label,
                zorder=3)


C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
NN, GR = "#1f77b4", "#ff7f0e"

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
axes = axes.ravel()

# -- panel 0: PINN loss components, neural field -------------------------------
tp = None
for i, (key, lab) in enumerate((("bfgs_loss", r"$\ell$"),
                                ("bfgs_L_res", r"$\ell_{\mathrm{pde}}$"),
                                ("bfgs_L_ic", r"$\ell_{\mathrm{ic}}$"),
                                ("bfgs_L_bc", r"$\ell_{\mathrm{bc}}$"),
                                ("bfgs_L_data", r"$\ell_{\mathrm{data}}$"))):
    Y, t, _ = series("pinn_pinn_nn_s*.npz", key)
    tp = t if tp is None else tp
    if key == "bfgs_loss":
        band(axes[0], Y, C[i], label=lab)
    else:
        axes[0].semilogy(np.arange(1, Y.shape[1] + 1), np.median(Y, axis=0),
                         color=C[i], lw=2.0, label=lab)
axes[0].set_title(f"PINN losses, NN ({tp.mean():.0f}$\\pm${tp.std(ddof=1):.0f} s, "
                  f"5 seeds)")
axes[0].set_ylabel("loss")
axes[0].legend(ncol=2, fontsize=9)

# -- panel 1: adjoint objective, both parameterizations ------------------------
Ynn, tnn, _ = series("adj_adjoint_nn_s*.npz", "__iter__", "loss")
Ygr, tgr, _ = series("adj_adjoint_grid_s*.npz", "__iter__", "loss")
band(axes[1], Ynn, NN,
     label=f"NN ({tnn.mean():.0f}$\\pm${tnn.std(ddof=1):.0f} s, 5 seeds)")
band(axes[1], Ygr, GR, ls="--",
     label=f"Coarse mesh ({tgr.mean():.0f} s, deterministic)")
axes[1].set_title("Adjoint objectives")
axes[1].set_ylabel("loss")
axes[1].legend(fontsize=9)

# -- panel 2: PINN forcing error, both parameterizations -----------------------
Pnn, _, _ = series("pinn_pinn_nn_s*.npz", "bfgs_rel_l2_f")
Pgr, tpg, _ = series("pinn_pinn_grid_s*.npz", "bfgs_rel_l2_f")
band(axes[2], Pnn, NN, label="NN, 5 seeds")
band(axes[2], Pgr, GR, ls="--", label="Coarse mesh, 5 seeds")
axes[2].set_title(r"Rel $L^2$ error of PINN forcing")
axes[2].set_ylabel(r"$\varepsilon_f$")
axes[2].legend(fontsize=9)

# -- panel 3: adjoint forcing error, both parameterizations --------------------
Ann, _, _ = series("adj_adjoint_nn_s*.npz", "__iter__", "rel_l2_f")
Agr, _, _ = series("adj_adjoint_grid_s*.npz", "__iter__", "rel_l2_f")
band(axes[3], Ann, NN, label="NN, 5 seeds")
band(axes[3], Agr, GR, ls="--", label="Coarse mesh, deterministic")
axes[3].set_title(r"Rel $L^2$ error of adjoint forcing")
axes[3].set_ylabel(r"$\varepsilon_f$")
axes[3].legend(fontsize=9)

# shared y-range on the two error panels so the factorial reads across them
pos = np.concatenate([v[v > TINY].ravel() for v in (Pnn, Pgr, Ann, Agr)])
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

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170)
print("wrote", OUT)
for nm, Y in (("PINN  NN", Pnn), ("PINN  grid", Pgr), ("adj   NN", Ann),
              ("adj   grid", Agr)):
    f = Y[:, -1]
    print(f"  {nm:11s} final eps_f {f.mean():.3e}"
          + (f" +- {f.std(ddof=1):.1e}" if f.size > 1 else "  (single run)"))
