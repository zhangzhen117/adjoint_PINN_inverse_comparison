"""Grid-convergence figure for the cylinder observations used in Test 4.

Three panels, left to right, in increasing order of how directly they bear on the
question "is the data the PINN is fitted to converged?":

  1. limit-cycle invariants -- error against the Richardson-extrapolated value
  2. mean-flow symmetry error -- truth-free: the exact limit cycle is symmetric
     about y = 0, so the mirror asymmetry of the time-mean is the error itself
  3. the observation vector -- successive-mesh differences, phase matched

The FEM production mesh that generated the paper's data is marked on each panel
at its equivalent cell count, so the reader can see where it sits in the family.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette reused from analysis/plot_ac_paper_history.py, where the adjacent-pair
# CVD check was run (worst CVD dE 9.1, worst normal-vision dE 19.6).
C = {"st": "#2a78d6", "cd": "#eb6834", "cl": "#1baf7a", "fem": "#e87ba4",
     "sym": "#2a78d6", "obs": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

CELLS = np.array([7616, 17136, 38556, 86913, 195432], float)
H = np.sqrt(77.215 / CELLS)                       # 2D: sqrt(area / cells)
LBL = ["L1", "L2", "L3", "L4", "L5"]

# from grid_report.txt
ST = np.array([0.18981, 0.19353, 0.19520, 0.19593, 0.19625])
CD = np.array([1.69820, 1.70217, 1.70396, 1.70467, 1.70504])
CL = np.array([0.46345, 0.46609, 0.46746, 0.46798, 0.46832])
ST_EXT, CD_EXT, CL_EXT = 0.1965256, 1.70548, 0.4688863
SYM = np.array([4.7947, 3.0515, 2.0097, 0.1537, 0.0822])          # %
OBS_D = np.array([7.0009, 1.8225, 2.8694, 0.7732])                # % , pairs 1-2..4-5

# FEM production mesh: 49490 P2 velocity nodes. A P2 triangle carries ~4 linear
# cells' worth of resolution, so the fair abscissa is the node count, not the
# element count; plotted as a band to keep that caveat visible.
FEM_N = 49490.0
FEM_H = np.sqrt(77.215 / FEM_N)
FEM_ST, FEM_CD, FEM_CL = 0.19656, 1.70372, 0.46708
FEM_SYM = 0.8573
FEM_OBS = 0.423

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
fig.patch.set_facecolor("#fcfcfb")


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor("#fcfcfb")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


# -- panel 1: invariants ------------------------------------------------------
ax = axes[0]
for y, ext, k, lab in ((ST, ST_EXT, "st", "$St$"), (CD, CD_EXT, "cd", r"$\bar{C}_D$"),
                       (CL, CL_EXT, "cl", r"$C_L$ amp")):
    e = 100 * np.abs(y - ext) / abs(ext)
    ax.loglog(H, e, "o-", color=C[k], lw=2.0, ms=6, solid_capstyle="round", label=lab)
ref = np.array([H[0], H[-1]])
ax.loglog(ref, 3.4 * (ref / H[0]) ** 2, "--", color=MUTED, lw=1.2)
ax.annotate("2nd order", xy=(H[2], 3.4 * (H[2] / H[0]) ** 2), xytext=(6, 4),
            textcoords="offset points", color=MUTED, fontsize=8.5, rotation=32)
for y, ext, k in ((FEM_ST, ST_EXT, "st"), (FEM_CD, CD_EXT, "cd"), (FEM_CL, CL_EXT, "cl")):
    ax.loglog([FEM_H], [100 * abs(y - ext) / abs(ext)], "*", color=C["fem"],
              ms=15, mec=INK, mew=0.5, zorder=5)
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
style(ax, r"cell size $h=\sqrt{A/N}$", "error vs extrapolated [%]",
      "limit-cycle invariants")

# -- panel 2: truth-free symmetry error ---------------------------------------
ax = axes[1]
ax.loglog(H, SYM, "o-", color=C["sym"], lw=2.0, ms=6, solid_capstyle="round")
ax.loglog([FEM_H], [FEM_SYM], "*", color=C["fem"], ms=17, mec=INK, mew=0.5, zorder=5)
ax.annotate("FEM production mesh\n(the paper's data)", xy=(FEM_H, FEM_SYM),
            xytext=(18, -30), textcoords="offset points", color=C["fem"],
            fontsize=9, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=C["fem"], lw=1.0))
for h, s, l in zip(H, SYM, LBL):
    ax.annotate(l, xy=(h, s), xytext=(0, 10), textcoords="offset points",
                color=MUTED, fontsize=8, ha="center")
ax.set_ylim(top=SYM.max() * 1.9)          # headroom so the L1 label clears the title
style(ax, r"cell size $h=\sqrt{A/N}$", "mirror asymmetry of the mean [%]",
      "mean-flow symmetry error (exact value: 0)")

# -- panel 3: the observation vector ------------------------------------------
ax = axes[2]
mid = np.sqrt(H[:-1] * H[1:])                       # difference between two levels
ax.loglog(mid, OBS_D, "o-", color=C["obs"], lw=2.0, ms=6, solid_capstyle="round",
          label="successive meshes")
ax.loglog([FEM_H], [FEM_OBS], "*", color=C["fem"], ms=17, mec=INK, mew=0.5,
          zorder=5, label="FEM vs finest")
ax.axhline(0.52, color=MUTED, lw=1.2, ls=":")
ax.annotate(r"published $\varepsilon_\nu=5.2\times10^{-3}$", xy=(mid[0], 0.52),
            xytext=(0, -13), textcoords="offset points", color=MUTED, fontsize=8.5)
ax.axhline(8.78, color=MUTED, lw=1.2, ls="-.")
ax.annotate(r"Table 2 $\varepsilon_\nu=8.8\times10^{-2}$", xy=(mid[0], 8.78),
            xytext=(0, 5), textcoords="offset points", color=MUTED, fontsize=8.5)
ax.legend(frameon=False, fontsize=8.5, loc="center left", labelcolor=INK)
ax.annotate("FEM data vs\nfinest mesh: 0.42%", xy=(FEM_H, FEM_OBS),
            xytext=(-14, -30), textcoords="offset points", color=C["fem"],
            fontsize=9, fontweight="bold", ha="right",
            arrowprops=dict(arrowstyle="-", color=C["fem"], lw=1.0))
style(ax, r"cell size $h=\sqrt{A/N}$", "relative difference [%]",
      "observation vector, phase matched")

fig.suptitle("Cylinder observations: grid convergence (OpenFOAM v2512, pimpleFoam, Re = 100)",
             fontsize=12.5, color=INK, x=0.012, ha="left", y=0.995)
fig.text(0.012, 0.925,
         "5 meshes, refinement ratio 1.5, 7 616 to 195 432 cells; time step verified "
         "independent (dt 0.005 vs 0.00125 shifts St by 4e-5 relative)",
         fontsize=9, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))

out = os.path.join(HERE, "cylinder_grid_convergence.png")
fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
print("wrote", out)
