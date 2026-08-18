"""Darcy field comparisons at gamma*, replacing Figures 4 and 5.

Same layout as the submitted figures: a 2x4 grid whose top row is the reference
field followed by the three reconstructions on a shared colour scale, and whose
bottom row carries the two colourbars in the first cell and the three pointwise
error maps, each titled with its relative error. The only change is the
regularization weight -- gamma* = 1e-2, the value Table 2 and Table 4 now use at
sigma = 1%, rather than the submitted 1e-3.

Figure geometry, fonts, colour maps and colourbar placement follow the original
plotting cell in cases/case2_darcy/darcy_noise.ipynb so the replacements sit beside the
untouched figures without looking different: 16x8 inches at font size 14, fields
interpolated onto a 100x100 grid and contoured with 50 levels in jet, errors in
seismic, and the two colourbars as narrow axes in the lower-left cell.

m is piecewise constant on triangles and u is nodal, so each is interpolated from
its own geometry -- element centroids for the log-permeability, mesh nodes for the
state.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "results", "darcy_field_data.npz")
OUTDIR = os.path.join(REPO, "debug")

d = np.load(DATA)
cent, nodes = d["centroids"], d["nodes"]
obs = d["obs_points"]

N = 100
xg = np.linspace(0, 1, N)
Xg, Yg = np.meshgrid(xg, xg)

METHODS = [("adjoint", "Adjoint"), ("pinn", "PINN"), ("eki", "EKI")]


def to_grid(pts, vals):
    """Interpolate a field onto the plotting grid, as the original cell did."""
    return np.nan_to_num(griddata(pts, vals, (Xg, Yg), method="linear"))


def panel(kind, sym, ref_key, pts, out_name):
    # field arrays use kind ("m"/"u"); the error scalars use the symbol ("f"/"u")
    ref = to_grid(pts, d[ref_key])
    recs = [to_grid(pts, d[f"{kind}_{k}"]) for k, _ in METHODS]
    errs = [r - ref for r in recs]
    eps = [float(d[f"eps_{sym}_{k}"]) for k, _ in METHODS]

    vmin = min(ref.min(), *[r.min() for r in recs])
    vmax = max(ref.max(), *[r.max() for r in recs])
    emax = max(np.abs(e).max() for e in errs)

    with plt.rc_context({"font.size": 14}):
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))

        for a, data, title in zip(ax[0], [ref] + recs,
                                  [f"Reference ${sym}$"] +
                                  [f"{lab} ${sym}$" for _, lab in METHODS]):
            im_f = a.contourf(Xg, Yg, data, levels=50, cmap="jet",
                              vmin=vmin, vmax=vmax)
        ax[0, 0].scatter(obs[:, 0], obs[:, 1], c="lime", s=20, marker="o",
                         edgecolors="black", linewidths=0.5)
        for a, title in zip(ax[0], [f"Reference ${sym}$"] +
                            [f"{lab} ${sym}$" for _, lab in METHODS]):
            a.set_title(title)

        ax[1, 0].axis("off")
        for a, e, (_, lab), r in zip(ax[1, 1:], errs, METHODS, eps):
            im_e = a.contourf(Xg, Yg, e, levels=50, cmap="seismic",
                              vmin=-emax, vmax=emax)
            a.set_title(rf"{lab} Error ($\varepsilon_{sym}$={r*100:.2f}%)")

        for a in ax.ravel():
            if a is not ax[1, 0]:
                a.set_xlabel("x")
                a.set_ylabel("y")
                a.set_aspect("equal")

        # aspect='equal' shrinks each axes inside its slot, so at the default
        # spacing the y-label of one panel lands on top of its left neighbour
        fig.subplots_adjust(wspace=0.45, hspace=0.30)

        # The original placed these at absolute figure coordinates 0.05 and 0.15,
        # which sat inside the lower-left cell under its default spacing. With the
        # wspace above they no longer line up, so they are positioned relative to
        # that cell at the same proportions the original produced.
        b = ax[1, 0].get_position()
        for frac, mappable, lab in ((0.04, im_f, f"${sym}$"), (0.66, im_e, "Error")):
            cax = fig.add_axes([b.x0 + frac * b.width, b.y0 + 0.05 * b.height,
                                0.067 * b.width, 0.85 * b.height])
            cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
            cb.set_label(lab, rotation=270, labelpad=20)

        out = os.path.join(OUTDIR, out_name)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    for (_, lab), e in zip(METHODS, eps):
        print(f"  {lab:8s} eps_{sym} = {100*e:.2f}%")


os.makedirs(OUTDIR, exist_ok=True)
print(f"sigma = {float(d['noise'])*100:g}%, gamma = {float(d['gamma']):g}, "
      f"seed {int(d['seed'])}\n")
panel("m", "f", "m_true", cent, "D_three_method_comparison_new.png")
print()
panel("u", "u", "u_true", nodes, "D_three_method_solution_comparison_new.png")
