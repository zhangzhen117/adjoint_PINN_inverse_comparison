"""Re-simulated state error for the bundle-G runs, matching Table 2's definition.

Table 2's eps_u for Test 4 is not the PINN's own field: the manuscript says it
"passes the recovered nu back through the forward solver for a fair cross-method
comparison". So for each run this marches the FEM production solver at that run's
nu_rec, from the same initial state the run was given, and compares the terminal
velocity to that arm's reference terminal field.

One asymmetry to keep in view when reading the OpenFOAM arm. Its reference
terminal field comes from OpenFOAM L5, but the re-simulation is done with the FEM
solver, so the comparison carries the FEM production mesh's own discretization
error (eps_u^solver = 1.4e-3) as a floor that the FEM arm does not pay. The OF
arm's eps_u therefore cannot go below roughly that level no matter how good nu is,
and a like-for-like reading should compare each arm against its own floor.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
from scipy.sparse import bmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "cylinder"))

from cylinder_solver import CylinderConfig, create_solver   # noqa: E402

T_WIN, DT = 5.0, 0.005
SRC = {"fem": "saturated.npz", "of": "saturated_of.npz"}


def main():
    mesh, asm, solver = create_solver(
        CylinderConfig(h_cyl=0.04, h_wake=0.08, h_far=0.5, dt=DT))
    M = asm.assemble_mass_p2()
    Mu = bmat([[M, None], [None, M]], format="csr")
    nps = mesh.n_p2

    def rel_l2_M(a, b):                       # mass-matrix L2, the honest norm
        d = a - b
        return float(np.sqrt((d @ (Mu @ d)) / (b @ (Mu @ b))))

    def rel_l2_plain(a, b):                   # nodal 2-norm, as rel_l2_T uses
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    out = {}
    for src, satname in SRC.items():
        z = np.load(os.path.join(REPO, "cylinder", "history", satname),
                    allow_pickle=True)
        u0 = np.asarray(z["u0"])
        times = np.asarray(z["times"])
        t0a = float(z["t0a"])
        i1 = int(np.argmin(np.abs(times - (t0a + T_WIN))))
        uref = np.asarray(z["snaps"][i1])
        assert abs(times[i1] - (t0a + T_WIN)) < 1e-6, "no reference at t0a+T"

        rows = []
        for d in sorted(glob.glob(os.path.join(REPO, "results", "G_runs",
                                               f"cyl_{src}_s*"))):
            zz = np.load(os.path.join(d, "pinn_inv.npz"), allow_pickle=True)
            nu = float(zz["nu_rec"])
            r = solver.solve_forward(T=T_WIN, nu=nu, u0=u0, dt=DT, ramp_T=0.0,
                                     adaptive=False)
            uT = r["u_final"]
            rows.append((nu, rel_l2_M(uT, uref), rel_l2_plain(uT, uref)))
            print(f"  {src} {os.path.basename(d)}: nu={nu:.6f}  "
                  f"eps_u(M)={rows[-1][1]:.4e}  eps_u(nodal)={rows[-1][2]:.4e}",
                  flush=True)

        # floor: the same solve at the true nu, i.e. what remains when nu is exact
        r0 = solver.solve_forward(T=T_WIN, nu=0.01, u0=u0, dt=DT, ramp_T=0.0,
                                  adaptive=False)
        floor_M = rel_l2_M(r0["u_final"], uref)
        floor_p = rel_l2_plain(r0["u_final"], uref)
        out[src] = (np.array(rows), floor_M, floor_p)
        print(f"  {src} FLOOR at nu=nu_true: eps_u(M)={floor_M:.4e}  "
              f"nodal={floor_p:.4e}\n", flush=True)

    print("=" * 68)
    for src, lab in (("of", "OpenFOAM L5 data (new)"), ("fem", "FEM data (published)")):
        a, fM, fp = out[src]
        print(f"{lab}")
        print(f"   eps_nu           : {np.abs(a[:,0]/0.01-1).mean():.3e} +- "
              f"{np.abs(a[:,0]/0.01-1).std(ddof=1):.3e}")
        print(f"   eps_u re-sim (M) : {a[:,1].mean():.3e} +- {a[:,1].std(ddof=1):.3e}")
        print(f"   eps_u re-sim (n) : {a[:,2].mean():.3e} +- {a[:,2].std(ddof=1):.3e}")
        print(f"   floor at nu_true : {fM:.3e} (M)   {fp:.3e} (nodal)")
    np.savez(os.path.join(HERE, "resim_eps_u.npz"),
             **{f"{k}_rows": v[0] for k, v in out.items()},
             **{f"{k}_floor": v[1] for k, v in out.items()})


if __name__ == "__main__":
    main()
