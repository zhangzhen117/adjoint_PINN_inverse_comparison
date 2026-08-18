"""Consolidate the Test 4 convergence histories into one file for plotting.

The published figure (C_optimization_history_4panel.png) shows a single PINN run
and a single adjoint run, both fitted to the finite-element observations. Test 4
is now five PINN seeds and five adjoint restarts, all fitted to the finite-volume
observations, so the replacement figure needs a different input: distributions
rather than single traces.

What this writes, into cylinder_history_data.npz:

  pinn_*        (5, n)  loss and its four components, state error, nu, per seed
  pinn_it       (n,)    iteration index, shared
  pinn_switch   scalar  iteration at which Adam hands over to SSBroyden
  adj_cold_*    (m,)    objective, nu, |grad| per objective evaluation
  adj_*_it              indices of the accepted BFGS iterates within those
  adj_warm_*    list    the same for each of the five restarts
  *_seconds             wall-clock axes where they are meaningful

On the time axes. The adjoint's cost per objective evaluation is uniform -- one
1000-step forward sweep plus its backward pass -- so runtime/nfev gives a real
wall-clock axis. The PINN's is not: Adam steps and quasi-Newton steps differ in
cost by more than an order of magnitude, and only the run total was recorded. The
PINN therefore gets its iteration axis plus a recorded total, and the two phases
are marked; inventing a per-step time for it would be fabrication.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "cylinder_history_data.npz")


def accepted_iterates(z):
    """Indices of the accepted BFGS iterates within the per-evaluation history.

    The inversion records one entry per objective call and scipy reports only a
    scalar iteration count, but the text and the figure quote iterations ("four
    BFGS iterations, nine function evaluations"). Replaying the recorded (J, g)
    stream through the identical minimize() call reproduces the run exactly --
    BFGS is a deterministic function of the values it is handed -- so its
    callback marks which evaluations were accepted. The reproduced nfev and nit
    are asserted against the recorded ones, and the callback misses the final
    iteration (the loop exits on the gradient test before it fires), so the
    returned point is appended.
    """
    th, J, g = (np.asarray(z[k], float) for k in ("theta", "J", "g"))
    n, idx = [0], []

    def obj(x):
        i = n[0]
        n[0] += 1
        assert abs(float(x[0]) - th[i]) < 1e-9, (i, float(x[0]), th[i])
        return J[i], np.array([g[i]])

    def cb(xk):
        idx.append(int(np.argmin(np.abs(th - float(xk[0])))))

    # options as in CylinderRunConfig: inv_maxiter, inv_gtol, inv_method_bfgs
    res = minimize(obj, np.array([th[0]]), method="BFGS", jac=True, callback=cb,
                   options={"maxiter": 10, "gtol": 1e-5, "disp": False,
                            "method_bfgs": "BFGS"})
    last = int(np.argmin(np.abs(th - float(res.x[0]))))
    it = [0] + idx + ([last] if last not in idx else [])
    assert n[0] == int(z["nfev"]), (n[0], int(z["nfev"]))
    assert int(res.nit) == int(z["nit"]) == len(it) - 1
    return np.array(it, int)


def main():
    d = {}

    # ------------------------------------------------------------ PINN, 5 seeds
    runs = sorted(glob.glob(os.path.join(REPO, "results", "G_runs", "cyl_of_s*",
                                         "pinn_inv.npz")))
    assert runs, "no bundle-G OpenFOAM PINN runs found"
    keys = ("loss", "pde", "ic", "bc", "data", "err", "nu")
    stacks = {k: [] for k in keys}
    times, switch = [], None
    for f in runs:
        z = np.load(f, allow_pickle=True)
        n = min(len(np.asarray(z[k])) for k in keys)
        for k in keys:
            stacks[k].append(np.asarray(z[k], float)[:n])
        times.append(float(z["runtime_sec"]))
        it = np.asarray(z["it"])[:n]
        ph = np.asarray(z["phase"])[:n]
        ch = np.where(ph[1:] != ph[:-1])[0]
        if switch is None and ch.size:
            switch = int(it[ch[0] + 1])
    m = min(len(v[0]) for v in stacks.values())
    for k in keys:
        d[f"pinn_{k}"] = np.stack([a[:m] for a in stacks[k]])
    d["pinn_it"] = it[:m]
    d["pinn_switch"] = switch
    d["pinn_seconds_total"] = np.array(times)
    d["pinn_seeds"] = np.array([os.path.basename(os.path.dirname(f)) for f in runs])
    print(f"PINN: {len(runs)} seeds x {m} records, Adam->SSBroyden at it={switch}, "
          f"t = {np.mean(times):.0f}+-{np.std(times, ddof=1):.0f}s")

    # ------------------------------------------------------- adjoint, cold start
    cold = os.path.join(HERE, "inverse_adjoint_of.npz")
    if os.path.exists(cold):
        z = np.load(cold, allow_pickle=True)
        J = np.asarray(z["J"], float)
        d["adj_cold_J"] = J
        d["adj_cold_nu"] = np.asarray(z["nu"], float)
        d["adj_cold_g"] = np.abs(np.asarray(z["g"], float))
        d["adj_cold_rel"] = np.asarray(z["rel"], float)
        d["adj_cold_t"] = float(z["runtime_sec"])
        # uniform cost per evaluation, so this axis is real rather than assumed
        d["adj_cold_seconds"] = (np.arange(1, len(J) + 1)
                                 * float(z["runtime_sec"]) / max(len(J), 1))
        d["adj_cold_it"] = accepted_iterates(z)
        print(f"adjoint cold: {int(z['nit'])} it / {len(J)} evals, "
              f"nu -> {float(z['nu_rec']):.8f}, "
              f"eps_nu = {float(z['rel_rec']):.3e}, {float(z['runtime_sec']):.0f}s")
    else:
        print("adjoint cold: NOT YET AVAILABLE")

    # --------------------------------------------------------- adjoint, restarts
    warm = sorted(glob.glob(os.path.join(HERE,
                                         "inverse_adjoint_of_restart_*.npz")))
    if warm:
        for i, f in enumerate(warm):
            z = np.load(f, allow_pickle=True)
            J = np.asarray(z["J"], float)
            d[f"adj_warm{i}_J"] = J
            d[f"adj_warm{i}_nu"] = np.asarray(z["nu"], float)
            d[f"adj_warm{i}_rel"] = np.asarray(z["rel"], float)
            d[f"adj_warm{i}_t"] = float(z["runtime_sec"])
            d[f"adj_warm{i}_nu0"] = float(z["nu0"])
            d[f"adj_warm{i}_seconds"] = (np.arange(1, len(J) + 1)
                                         * float(z["runtime_sec"]) / max(len(J), 1))
            d[f"adj_warm{i}_it"] = accepted_iterates(z)
            print(f"adjoint restart {i}: nu0={float(z['nu0']):.6f} -> "
                  f"{float(z['nu_rec']):.8f}, eps_nu={float(z['rel_rec']):.3e}, "
                  f"{int(z['nit'])} it / {len(J)} evals, "
                  f"{float(z['runtime_sec']):.0f}s")
        d["adj_n_warm"] = len(warm)
    else:
        print("adjoint restarts: NOT YET AVAILABLE")

    d["nu_true"] = 0.01
    d["source"] = ("Test 4 fitted to OpenFOAM L5 observations; PINN = bundle G "
                   "(paper setup, published budget), adjoint = case4_cylinder/gridstudy")
    np.savez(OUT, **d)
    print(f"\nwrote {OUT}  ({len(d)} arrays)")


if __name__ == "__main__":
    main()
