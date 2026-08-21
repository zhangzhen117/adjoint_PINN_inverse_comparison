"""Test 4 convergence histories, replacing C_optimization_history_4panel.png.

Same four panels as the published figure -- PINN losses, PINN viscosity, adjoint
objectives, adjoint viscosities -- but the content underneath has changed twice
over, so the figure has to as well:

* Test 4 is now five PINN initializations and five adjoint restarts rather than
  one run of each, and the seed spread is the headline finding of that benchmark
  (the viscosity scatters by a factor of ten while the flow reconstruction does
  not). Single traces would hide exactly what the section reports.
* Both estimators are now fitted to the grid-converged finite-volume
  observations, so the adjoint objective can no longer reach zero: it flattens at
  the mismatch between the two discretizations. That floor is the point of the
  top-right panel and is annotated rather than left for the reader to notice.
* The adjoint history holds one record per objective call, and drawing it against
  a "Iteration" axis conflicted with the text, which reports four BFGS iterations
  against nine function evaluations. The adjoint panels now plot the accepted
  iterates only (indices supplied by make_history_data.py), so their axis means
  what it says; the line-search trials in between are not iterates.

Per-seed traces are drawn thin and the median bold. The loss panel shows the
median of each component, with the composite loss also drawn per seed, since five
components times five seeds would be unreadable.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import figformat
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "cases", "case4_cylinder", "gridstudy",
                    "cylinder_history_data.npz")
# Written straight into the manuscript checkout: one script per figure,
# no "_new" copy to diverge from what the paper actually shows.
OUT = os.path.join(REPO, "paper_overleaf", "C_optimization_history_4panel.png")

# the published figure's colour order, kept so the two read as the same family
C_LOSS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
C_COLD, C_WARM = "#1f77b4", "#ff7f0e"
NU_TRUE = 0.01
T_PINN = "196$\\pm$6"          # Table 2(b); the histories record 191$\pm$7 s of
                                # training, the rest is setup

d = np.load(DATA, allow_pickle=True)
it = d["pinn_it"]
switch = int(d["pinn_switch"])

fig, ax = plt.subplots(2, 2, figsize=(19.5, 13.5))
plt.rcParams.update({"font.size": 20})
for a in ax.ravel():
    a.set_xlabel("Iteration", fontsize=20)
    a.grid(True, which="both", lw=0.5, alpha=0.35)
    a.tick_params(labelsize=17)
for a in ax[:, 1]:                       # adjoint panels: integer iteration axis
    a.set_xlabel("BFGS iteration", fontsize=20)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

# ---------------------------------------------------------------- PINN losses
a = ax[0, 0]
for k, (key, lab, col) in enumerate((("loss", r"$\ell$", C_LOSS[0]),
                                     ("pde", r"$\ell_{\rm pde}$", C_LOSS[1]),
                                     ("ic", r"$\ell_{\rm ic}$", C_LOSS[2]),
                                     ("bc", r"$\ell_{\rm bc}$", C_LOSS[3]),
                                     ("data", r"$\ell_{\rm data}$", C_LOSS[4]))):
    Y = d[f"pinn_{key}"]
    if key == "loss":                        # spread shown for the composite only
        for y in Y:
            a.semilogy(it, y, color=col, lw=0.8, alpha=0.30, zorder=2)
    a.semilogy(it, np.median(Y, axis=0), color=col, lw=2.4, label=lab, zorder=3)
a.axvline(switch, color="0.35", ls="--", lw=1.8)
a.annotate("Adam -> SSBroyden", xy=(switch, np.median(d["pinn_loss"], axis=0)[
    int(np.argmin(np.abs(it - switch)))]), xytext=(12, 10),
    textcoords="offset points", fontsize=18)
a.plot([switch], [np.median(d["pinn_loss"], axis=0)[
    int(np.argmin(np.abs(it - switch)))]], "ko", ms=9, zorder=4)
a.set_ylabel("loss", fontsize=20)
a.set_title(f"PINN losses (runtime {T_PINN} s, 5 seeds; thin = seeds)", fontsize=21)
a.legend(fontsize=17, ncol=2, loc="lower left")

# ------------------------------------------------------------- PINN viscosity
a = ax[1, 0]
NU = d["pinn_nu"]
for y in NU:
    a.semilogy(it, y, color=C_COLD, lw=0.9, alpha=0.35, zorder=2)
a.semilogy(it, np.median(NU, axis=0), color=C_COLD, lw=2.6, label=r"$\nu$ (median)",
           zorder=3)
a.axhline(NU_TRUE, color="r", ls="--", lw=2.0, label=r"$\nu_{\rm true}$")
a.axvline(switch, color="0.35", ls="--", lw=1.8)
a.annotate("Adam -> SSBroyden", xy=(switch, np.median(NU, axis=0)[
    int(np.argmin(np.abs(it - switch)))]), xytext=(12, 10),
    textcoords="offset points", fontsize=18)
eps = np.abs(NU[:, -1] / NU_TRUE - 1)
a.set_ylabel(r"$\nu$", fontsize=20)
a.set_title(f"PINN viscosity (rel err {eps.mean():.2e} $\\pm$ {eps.std(ddof=1):.1e})",
            fontsize=21)
a.legend(fontsize=17)

# -------------------------------------------------------- adjoint objectives
a = ax[0, 1]
kc = d["adj_cold_it"]                    # accepted iterates, not function calls
Jc = d["adj_cold_J"][kc]
a.semilogy(np.arange(len(kc)), Jc, color=C_COLD, lw=2.6,
           label=f"Adjoint ({d['adj_cold_t']:.0f} s, {len(kc)-1} it)",
           marker="o", ms=6)
nw = int(d["adj_n_warm"])
tw = np.array([float(d[f"adj_warm{i}_t"]) for i in range(nw)])
kw = [d[f"adj_warm{i}_it"] for i in range(nw)]
for i in range(nw):
    J = d[f"adj_warm{i}_J"][kw[i]]
    a.semilogy(np.arange(len(J)), J, color=C_WARM, lw=1.6, alpha=0.75,
               marker="s", ms=5,
               label=(f"Adjoint restart ({tw.mean():.0f}$\\pm${tw.std(ddof=1):.0f} s,"
                      f" {nw} seeds)" if i == 0 else None))
floor = min(Jc.min(), min(d[f"adj_warm{i}_J"][kw[i]].min() for i in range(nw)))
a.axhline(floor, color="0.35", ls=":", lw=2.0)
a.annotate(f"discretization mismatch floor  {floor:.2e}",
           xy=((len(kc) - 1) * 0.62, floor), xytext=(0, 14),
           textcoords="offset points", fontsize=17, color="0.25", ha="center")
a.set_ylabel("J", fontsize=20)
a.set_title("Adjoint objectives (finite-volume observations)", fontsize=21)
a.legend(fontsize=17)

# ------------------------------------------------------- adjoint viscosities
a = ax[1, 1]
nu_c = d["adj_cold_nu"][kc]
ec = abs(nu_c[-1] / NU_TRUE - 1)
a.semilogy(np.arange(len(nu_c)), nu_c, color=C_COLD, lw=2.6, marker="o",
           ms=6, label=f"Adjoint $\\nu$ (rel err {ec:.3e})")
ew = np.array([abs(d[f"adj_warm{i}_nu"][-1] / NU_TRUE - 1) for i in range(nw)])
for i in range(nw):
    nu = d[f"adj_warm{i}_nu"][kw[i]]
    a.semilogy(np.arange(len(nu)), nu, color=C_WARM, lw=1.6, alpha=0.75,
               marker="s", ms=5,
               label=(f"Adjoint restart $\\nu$ (rel err {ew.mean():.3e})"
                      if i == 0 else None))
a.axhline(NU_TRUE, color="r", ls="--", lw=2.0, label=r"$\nu_{\rm true}$")
a.set_ylabel(r"$\nu$", fontsize=20)
a.set_title("Adjoint viscosities (all restarts reach the cold-start value)",
            fontsize=21)
a.legend(fontsize=16)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(figformat.target(OUT), dpi=figformat.dpi(140))
print("wrote", figformat.target(OUT))
print(f"  PINN  eps_nu {eps.mean():.3e} +- {eps.std(ddof=1):.1e}")
print(f"  adj   cold {ec:.4e}   restarts {ew.mean():.4e} +- {ew.std(ddof=1):.1e}")
print(f"  objective floor {floor:.3e}")
