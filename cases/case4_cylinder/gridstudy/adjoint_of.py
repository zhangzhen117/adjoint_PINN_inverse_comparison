"""Cylinder adjoint inversion against the converged OpenFOAM L5 observations.

The published Test 4 adjoint inverts through the same FEM discretization that
generated its observations, so the discretization error cancels identically and it
recovers nu to 3.99e-8 -- a measure of optimizer convergence rather than physical
accuracy. Fitting it to the finite-volume observations instead breaks that
cancellation: the forward map is still the FEM solver, but the data is no longer
its own output, so nu must now absorb the discretization mismatch. This puts the
adjoint and the PINN on the same observations in Table 2, and it is what the
adjoint's accuracy on this benchmark looks like without the inverse crime.

Runs, all against saturated_of.npz / probe_obs_of.npz:

  * cold start from nu0 = 100 nu_true, matching the published protocol;
  * one restart per PINN seed, warm started from that seed's recovered nu, so the
    hybrid claim rests on a distribution rather than on one favourable estimate.

eps_u is the re-simulated state error in the production mass-matrix L2 norm,
measured against the finite-volume terminal field, which is the same definition
Table 2 uses for every other entry.
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
from scipy.sparse import bmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "cases", "case4_cylinder"))

from cylinder_config import CylinderRunConfig          # noqa: E402
from cylinder_api import CylinderAPI                   # noqa: E402
from cylinder_solver import CylinderConfig, create_solver  # noqa: E402

HIST = os.path.join(REPO, "cases", "case4_cylinder", "history")
T_WIN, DT, NU_TRUE = 5.0, 0.005, 0.01


def base_cfg():
    c = CylinderRunConfig()
    c.saturated_path = os.path.join(HIST, "saturated_of.npz")
    c.obs_path = os.path.join(HIST, "probe_obs_of.npz")
    c.hist_dir = HERE
    return c


def main():
    out = {}

    # ---------------------------------------------------------- cold start
    cfg = base_cfg()
    print("=" * 66 + "\n=== cold adjoint on finite-volume observations ===\n"
          + "=" * 66, flush=True)
    api = CylinderAPI(cfg)
    cold = api.invert_adjoint(
        verbose=True, save_path=os.path.join(HERE, "inverse_adjoint_of.npz"))
    out["cold"] = cold
    print(f"--> cold: nu={cold['nu_rec']:.8f}  eps_nu={cold['rel_rec']:.4e}  "
          f"nit={cold['nit']} nfev={cold['nfev']}  {cold['runtime_sec']:.0f}s",
          flush=True)

    # ------------------------------------------------- warm starts, per seed
    seeds = sorted(glob.glob(os.path.join(REPO, "results", "G_runs", "cyl_of_s*")))
    warm = []
    for d in seeds:
        z = np.load(os.path.join(d, "pinn_inv.npz"), allow_pickle=True)
        nu0 = float(z["nu_rec"])
        tag = os.path.basename(d)
        print(f"\n--- restart from {tag}: nu0={nu0:.8f} "
              f"(eps_nu={abs(nu0/NU_TRUE-1):.3e}) ---", flush=True)
        r = CylinderAPI(base_cfg()).invert_adjoint(
            verbose=False, nu0=nu0,
            save_path=os.path.join(HERE, f"inverse_adjoint_of_restart_{tag}.npz"))
        r["tag"] = tag
        r["nu0_eps"] = abs(nu0 / NU_TRUE - 1)
        warm.append(r)
        print(f"--> {tag}: nu={r['nu_rec']:.8f}  eps_nu={r['rel_rec']:.4e}  "
              f"nit={r['nit']} nfev={r['nfev']}  {r['runtime_sec']:.0f}s", flush=True)
    out["warm"] = warm

    # ------------------------------------------------ eps_u by re-simulation
    print("\n=== re-simulated state error against the finite-volume reference ===",
          flush=True)
    mesh, asm, solver = create_solver(
        CylinderConfig(h_cyl=0.04, h_wake=0.08, h_far=0.5, dt=DT))
    M = asm.assemble_mass_p2()
    Mu = bmat([[M, None], [None, M]], format="csr")
    z = np.load(os.path.join(HIST, "saturated_of.npz"), allow_pickle=True)
    u0 = np.asarray(z["u0"])
    uref = np.asarray(z["snaps"][1])

    def eps_u(nu):
        r = solver.solve_forward(T=T_WIN, nu=nu, u0=u0, dt=DT, ramp_T=0.0,
                                 adaptive=False)
        d = r["u_final"] - uref
        return float(np.sqrt((d @ (Mu @ d)) / (uref @ (Mu @ uref))))

    eu_cold = eps_u(cold["nu_rec"])
    eu_floor = eps_u(NU_TRUE)
    print(f"  cold  eps_u = {eu_cold:.4e}", flush=True)
    print(f"  floor eps_u = {eu_floor:.4e}  (at nu = nu_true)", flush=True)

    nu_w = np.array([r["nu_rec"] for r in warm])
    e_w = np.array([r["rel_rec"] for r in warm])
    t_w = np.array([r["runtime_sec"] for r in warm])
    print("\n" + "=" * 66)
    print("TABLE 2, Test 4, adjoint on finite-volume observations")
    print("=" * 66)
    print(f"  eps_nu (cold)      = {cold['rel_rec']:.4e}")
    print(f"  eps_u  (cold)      = {eu_cold:.4e}    floor {eu_floor:.4e}")
    print(f"  t      (cold)      = {cold['runtime_sec']:.0f} s  "
          f"(nit {cold['nit']}, nfev {cold['nfev']})")
    if len(warm):
        print(f"  restart eps_nu     = {e_w.mean():.4e} +- {e_w.std(ddof=1):.2e}"
              f"   [{e_w.min():.2e}, {e_w.max():.2e}]")
        print(f"  restart t          = {t_w.mean():.0f} +- {t_w.std(ddof=1):.0f} s")
        print(f"  restart nit/nfev   = "
              + ", ".join(f"{r['nit']}/{r['nfev']}" for r in warm))
    np.savez(os.path.join(HERE, "adjoint_of_summary.npz"),
             nu_cold=cold["nu_rec"], eps_nu_cold=cold["rel_rec"],
             eps_u_cold=eu_cold, eps_u_floor=eu_floor,
             t_cold=cold["runtime_sec"], nit_cold=cold["nit"],
             nfev_cold=cold["nfev"], nu_warm=nu_w, eps_nu_warm=e_w, t_warm=t_w)


if __name__ == "__main__":
    main()
