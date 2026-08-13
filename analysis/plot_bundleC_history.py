"""Bundle C convergence histories: paper vs modern vs vanilla-Adam on the cylinder.

Two panels, both against **wall-clock**, because the three setups do very different
amounts of work per iteration -- an SSBroyden iteration is not comparable to an Adam
step -- so iteration count would misrepresent the cost. Iteration counts are printed
in the console summary instead.

Per-seed traces are drawn thin and translucent with the median bold on top, so the
seed spread (the point of R1.7/R3.5) and the oscillation are both visible rather
than averaged away.
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NU_TRUE = 0.01
PUBLISHED = 5.235e-3          # Table 2 cylinder PINN entry

# Categorical slots 1-3 of the validated reference palette (light mode).
# Validated all-pairs: worst CVD dE 9.2, worst normal-vision dE 24.0.
COLORS = {"paper": "#2a78d6", "paper_converged": "#4a3aa7",
          "modern": "#eb6834", "vanilla_adam": "#1baf7a"}
LABELS = {"paper": "paper setup, published budget",
          "paper_converged": "paper setup, run to plateau",
          "modern": "modern (ModifiedMLP+SOAP+gradnorm+PT)",
          "vanilla_adam": "vanilla PINN (tanh MLP, Adam only)"}
SHORT = {"paper": "paper", "paper_converged": "paper (converged)",
         "modern": "modern", "vanilla_adam": "vanilla Adam"}
ORDER = ("paper", "paper_converged", "modern", "vanilla_adam")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def load(case):
    """Return a list of (time_s, loss, eps_nu) per seed."""
    paths = sorted(glob.glob(os.path.join(REPO, f"results/C_runs/{case}_s*/pinn_inv.npz")))
    # paper_converged_s* would also match paper_s*; filter to the exact case.
    paths = [q for q in paths
             if os.path.basename(os.path.dirname(q)).rsplit("_s", 1)[0] == case]

    runs = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        it = np.asarray(d["it"], dtype=float)
        loss = np.asarray(d["loss"], dtype=float)
        nu = np.asarray(d["nu"], dtype=float)
        eps = np.abs(nu - NU_TRUE) / NU_TRUE
        runtime = float(d["runtime_sec"])
        # Steps are uniform in cost within a run, so iteration maps linearly to time.
        t = it / it.max() * runtime
        runs.append((t, loss, eps))
    return runs


def median_curve(runs, idx):
    """Median across seeds on a common time grid (curves have equal length here)."""
    n = min(len(r[0]) for r in runs)
    t = np.median(np.stack([r[0][:n] for r in runs]), axis=0)
    y = np.median(np.stack([r[idx][:n] for r in runs]), axis=0)
    return t, y


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.patch.set_facecolor("#fcfcfb")

panels = [(0, 1, "composite training loss", "loss"),
          (1, 2, r"viscosity error  $|\nu-\nu^\star|/\nu^\star$", "eps")]

summary = {}
for ax, (ai, idx, ylab, kind) in zip(axes, panels):
    ax.set_facecolor("#fcfcfb")
    for case in ORDER:
        runs = load(case)
        if not runs:
            continue
        c = COLORS[case]
        for t, loss, eps in runs:                       # per-seed, translucent
            ax.plot(t, (loss if idx == 1 else eps), color=c, lw=0.7, alpha=0.28,
                    solid_capstyle="round", zorder=2)
        tm, ym = median_curve(runs, idx)                # median, bold
        ax.plot(tm, ym, color=c, lw=2.0, solid_capstyle="round", zorder=3,
                label=LABELS[case])
        summary.setdefault(case, {})[kind] = (tm, ym, runs)

    if kind == "eps":
        ax.axhline(PUBLISHED, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(0.985, PUBLISHED * 1.22, "published 5.24e-3", transform=
                ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8.5, color=MUTED)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall-clock time [s]", fontsize=10, color=INK)
    ax.set_ylabel(ylab, fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

# Direct labels at each median's end point -- also the "relief" the palette
# validator requires for the aqua slot on a light surface.
for case in ORDER:
    if case not in summary:
        continue
    tm, ym, _ = summary[case]["eps"]
    axes[1].annotate(SHORT[case], xy=(tm[-1], ym[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     color=COLORS[case], fontsize=9, va="center", fontweight="bold")

# Mark where each arm's nu error is lowest -- the point the loss cannot identify.
for case in ORDER:
    if case not in summary:
        continue
    tm, ym, _ = summary[case]["eps"]
    i = int(np.argmin(ym))
    axes[1].plot([tm[i]], [ym[i]], marker="o", ms=7, color=COLORS[case],
                 mec="#fcfcfb", mew=1.6, zorder=5)

axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
axes[1].set_xlim(right=axes[1].get_xlim()[1] * 2.6)   # room for the direct labels

fig.suptitle("Cylinder inverse PINN: lower loss does not mean better viscosity",
             fontsize=12.5, color=INK, x=0.02, ha="left", y=0.99)
fig.text(0.02, 0.925,
         "thin = seeds, bold = median, dot = lowest nu error reached.  The loss falls "
         "monotonically in every arm while nu degrades: the composite loss is "
         "minimised away from the true viscosity.",
         fontsize=9, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))

out = os.path.join(REPO, "debug", "bundleC_history.png")
fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
print("wrote", out)

print(f"\n{'case':>14} {'seeds':>5} {'iters':>7} {'t [s]':>8} {'final eps':>11} "
      f"{'median eps':>11} {'oracle best':>12}")
for case in ORDER:
    runs = load(case)
    if not runs:
        continue
    fin = [r[2][-1] for r in runs]
    best = [r[2].min() for r in runs]
    print(f"{SHORT[case]:>14} {len(runs):>5} {len(runs[0][0]):>7} "
          f"{np.mean([r[0][-1] for r in runs]):>8.0f} "
          f"{np.median(fin):>11.3e} {np.median(fin):>11.3e} {np.median(best):>12.3e}")
