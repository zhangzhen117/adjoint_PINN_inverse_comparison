"""Test 3 convergence histories over five seeds, replacing AC_training_history.png.

Same four panels as the published figure -- PINN losses, adjoint objectives, and
the two reaction-error traces -- redrawn from five runs of each arm instead of one.
The PINN panels show the SSBroyden phase, as the published figure did; the Adam
warmup is not plotted.

The restart curves are the ones that changed most. The published figure showed a
single restart that ran 126 evaluations and stopped, against a cold start that ran
313, because that run was made with scipy_adj_nn_maxiter = 100 while the cold
baseline used 200 -- so the two were not run to the same rule and the restart
looked half as expensive as it is. Re-run at the current setting from each of the
five PINN estimates, the restarts take 282-303 evaluations, essentially the same
as cold, and reach a reaction error roughly nine times better. The figure now
shows that: the restart curves start low and end low, but they do not stop early.

Traces are truncated to the shortest run of each arm before the median and band
are taken, since the adjoint runs stop at different evaluation counts.
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "debug", "AC_training_history_new.png")

C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
C_COLD, C_WARM = "#1f77b4", "#ff7f0e"


def load_pinn():
    out, t = {}, []
    keys = ("loss", "L_res", "L_ic", "L_bc", "L_data", "rel_l2_f")
    stacks = {k: [] for k in keys}
    for f in sorted(glob.glob(os.path.join(REPO, "results", "D_runs", "ac_s*",
                                           "ac3d_pinn.npz"))):
        z = np.load(f, allow_pickle=True)
        for k in keys:
            stacks[k].append(np.asarray(z[f"bfgs_{k}"], float))
        t.append(float(z["runtime_sec"]) if "runtime_sec" in z.files else np.nan)
    n = min(a.size for a in stacks["loss"])
    for k in keys:
        out[k] = np.stack([a[:n] for a in stacks[k]])
    return out, n, np.array(t)


def load_adj(pattern, fname):
    L, E, t = [], [], []
    for f in sorted(glob.glob(os.path.join(REPO, "results", pattern, fname))):
        z = np.load(f, allow_pickle=True)
        ev = z["evals"]
        L.append(np.array([e["loss"] for e in ev], float))
        E.append(np.array([e["rel_l2_f"] for e in ev], float))
        t.append(float(z["runtime_sec"]))
    n = min(a.size for a in L)
    return (np.stack([a[:n] for a in L]), np.stack([a[:n] for a in E]),
            n, np.array(t))


P, n_p, t_p = load_pinn()
Lc, Ec, n_c, t_c = load_adj(os.path.join("D_runs", "ac_adj_s*"), "ac3d_adj.npz")
Lw, Ew, n_w, t_w = load_adj(os.path.join("H_runs", "ac_restart_s*"),
                            "ac3d_adj_restart.npz")

# bundle D records the PINN runtime in the sweep, not the npz
if not np.isfinite(t_p).all():
    import json
    D = [json.loads(l) for l in open(os.path.join(REPO, "results", "D.jsonl"))
         if l.strip().startswith("{")]
    t_p = np.array([r["runtime_s"] for r in D
                    if r["benchmark"] == "allencahn" and r["method"] == "pinn"])

fig, ax = plt.subplots(2, 2, figsize=(19, 13.5))
for a in ax.ravel():
    a.set_xlabel("Iteration", fontsize=20)
    a.grid(True, which="both", lw=0.5, alpha=0.35)
    a.tick_params(labelsize=17)


def band(a, x, Y, col, lw=2.4, label=None, alpha=0.30):
    for y in Y:
        a.semilogy(x, y, color=col, lw=0.8, alpha=alpha, zorder=2)
    a.semilogy(x, np.median(Y, axis=0), color=col, lw=lw, label=label, zorder=3)


# ---------------------------------------------------------------- PINN losses
a = ax[0, 0]
x = np.arange(n_p)
for key, lab, col in (("loss", r"$\ell$", C[0]), ("L_res", r"$\ell_{\rm pde}$", C[1]),
                      ("L_ic", r"$\ell_{\rm ic}$", C[2]),
                      ("L_bc", r"$\ell_{\rm bc}$", C[3]),
                      ("L_data", r"$\ell_{\rm data}$", C[4])):
    if key == "loss":
        band(a, x, P[key], col, label=lab)
    else:
        a.semilogy(x, np.median(P[key], axis=0), color=col, lw=2.2, label=lab)
a.set_ylabel("loss", fontsize=20)
a.set_title(f"PINN losses ({t_p.mean():.0f}$\\pm${t_p.std(ddof=1):.0f} s, 5 seeds)",
            fontsize=21)
a.legend(fontsize=17, ncol=2, loc="upper right")

# ------------------------------------------------------------ adjoint losses
a = ax[0, 1]
band(a, np.arange(1, n_c + 1), Lc, C_COLD,
     label=f"Adjoint ({t_c.mean():.0f}$\\pm${t_c.std(ddof=1):.0f} s)")
band(a, np.arange(1, n_w + 1), Lw, C_WARM,
     label=f"Adjoint restart ({t_w.mean():.0f}$\\pm${t_w.std(ddof=1):.0f} s)")
a.set_ylabel("loss", fontsize=20)
a.set_title("Adjoint loss histories (5 seeds each)", fontsize=21)
a.legend(fontsize=17)

# ------------------------------------------------------- reaction-term errors
a = ax[1, 0]
band(a, np.arange(n_p), P["rel_l2_f"], C_COLD, label="median")
a.fill_between(np.arange(n_p), P["rel_l2_f"].min(axis=0), P["rel_l2_f"].max(axis=0),
               color=C_COLD, alpha=0.13, zorder=1, label="seed range")
f = P["rel_l2_f"][:, -1]
a.set_ylabel(r"$\varepsilon_f$", fontsize=20)
a.set_title(f"Rel $L^2$ error of PINN forcing   "
            f"({f.mean():.2e}$\\pm${f.std(ddof=1):.1e})", fontsize=21)
a.set_ylim(1e-4, 1e1)
a.legend(fontsize=17)

a = ax[1, 1]
band(a, np.arange(1, n_c + 1), Ec, C_COLD, label="Adjoint")
band(a, np.arange(1, n_w + 1), Ew, C_WARM, label="Adjoint restart")
fc, fw = Ec[:, -1], Ew[:, -1]
a.set_ylabel(r"$\varepsilon_f$", fontsize=20)
a.set_title(f"Rel $L^2$ error of adjoint forcing", fontsize=21)
a.text(0.03, 0.06, f"cold {fc.mean():.2e},  restart {fw.mean():.2e}",
       transform=a.transAxes, fontsize=16, color="0.25")
a.set_ylim(1e-5, 1e1)
a.legend(fontsize=17)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
print(f"  PINN    eps_f {f.mean():.3e} +- {f.std(ddof=1):.1e}   t {t_p.mean():.0f}s")
print(f"  adjoint eps_f {fc.mean():.3e} +- {fc.std(ddof=1):.1e}  t {t_c.mean():.0f}s")
print(f"  restart eps_f {fw.mean():.3e} +- {fw.std(ddof=1):.1e}  t {t_w.mean():.0f}s")
print(f"  truncated to {n_p} PINN, {n_c} cold, {n_w} restart evaluations")
