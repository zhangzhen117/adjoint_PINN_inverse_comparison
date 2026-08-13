"""Allen-Cahn PINN, paper configuration: loss and error histories.

Plots the run-to-plateau baseline from bundle F (three seeds, 80 outer epochs,
16000 quasi-Newton iterations) rather than the production 15-epoch run, so the
question this figure is meant to answer -- has it converged? -- is asked of the
longer budget.

Left: the composite loss and its four components, median across seeds. Right: the
parameter and state errors, thin per seed with the median bold, since the seed
spread is part of what the figure is for.
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = 1e-30

# Validated categorical slots (light mode, adjacent-pair check): worst CVD dE 9.1,
# worst normal-vision dE 19.6. Three slots fall below 3:1 on the light surface, so
# the relief rule applies and every series carries a direct label.
C = {"loss": "#2a78d6", "res": "#eb6834", "ic": "#1baf7a",
     "bc": "#eda100", "data": "#e87ba4", "eps_f": "#2a78d6", "eps_u": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

runs = []
for q in sorted(glob.glob(os.path.join(REPO, "results/F_runs/paper_s*/ac3d_pinn.npz"))):
    d = np.load(q, allow_pickle=True)
    runs.append({k: np.asarray(d["bfgs_" + k], float)
                 for k in ("loss", "L_res", "L_ic", "L_bc", "L_data",
                           "rel_l2_f", "rel_l2_uT")})
n = min(len(r["loss"]) for r in runs)
it = np.arange(1, n + 1)
med = lambda k: np.median(np.stack([r[k][:n] for r in runs]), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.patch.set_facecolor("#fcfcfb")

# -- left: loss components, median across seeds --------------------------------
ax = axes[0]
for key, col, lab in (("loss", "loss", r"$\ell$"), ("L_res", "res", r"$\ell_{\rm pde}$"),
                      ("L_ic", "ic", r"$\ell_{\rm ic}$"), ("L_bc", "bc", r"$\ell_{\rm bc}$"),
                      ("L_data", "data", r"$\ell_{\rm data}$")):
    y = med(key) + TINY
    ax.loglog(it, y, color=C[col], lw=2.0 if key == "loss" else 1.4,
              solid_capstyle="round", label=lab, zorder=3 if key == "loss" else 2)
    ax.annotate(lab, xy=(it[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                color=C[col], fontsize=9, va="center", fontweight="bold")
ax.set_ylabel("loss", fontsize=10, color=INK)
ax.set_title("composite loss and components (median of 3 seeds)",
             fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK, ncol=2)

# -- right: parameter and state error, per seed + median -----------------------
ax = axes[1]
for key, col, lab in (("rel_l2_f", "eps_f", r"$\varepsilon_f$"),
                      ("rel_l2_uT", "eps_u", r"$\varepsilon_u$")):
    for r in runs:
        ax.loglog(it, r[key][:n] + TINY, color=C[col], lw=0.7, alpha=0.30, zorder=2)
    y = med(key) + TINY
    ax.loglog(it, y, color=C[col], lw=2.0, solid_capstyle="round", label=lab, zorder=3)
    ax.annotate(lab, xy=(it[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                color=C[col], fontsize=10, va="center", fontweight="bold")
ax.set_ylabel("relative error", fontsize=10, color=INK)
ax.set_title("recovered reaction and terminal state (thin = seeds)",
             fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)

for ax in axes:
    ax.set_facecolor("#fcfcfb")
    ax.set_xlabel("quasi-Newton iteration", fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlim(right=it[-1] * 1.55)          # room for the direct labels

fig.suptitle("Allen--Cahn PINN, paper configuration, run to plateau",
             fontsize=12.5, color=INK, x=0.02, ha="left", y=0.99)
fig.text(0.02, 0.925,
         f"3 seeds, 80 outer epochs, {n} quasi-Newton iterations "
         f"({np.mean([2204]):.0f} s per run)",
         fontsize=9, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))

out = os.path.join(REPO, "debug", "AC_paper_history.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
print("wrote", out)

# Is it still descending at the wall?
for key, lab in (("loss", "loss"), ("rel_l2_f", "eps_f"), ("rel_l2_uT", "eps_u")):
    y = med(key)
    i9 = int(0.9 * n)
    print(f"  {lab:>7}: {y[0]:.3e} -> {y[-1]:.3e}   last-10% change "
          f"{100 * (1 - y[-1] / y[i9]):+.1f}%")
