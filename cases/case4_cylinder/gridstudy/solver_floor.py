"""Discretization floor eps_u^solver for Test 4, matching Table 2's definition.

Table 2 defines eps_u^solver as "the relative L2 error of the forward solver
against a finer-mesh reference". Tests 1-3 have it; Test 4's cell is empty. This
computes it with the paper's own FEM solver, so the number is comparable with the
other three rows rather than borrowed from the OpenFOAM study.

Protocol, chosen to match what the inverse problem actually asks of the solver:

    both meshes start from the SAME saturated state u0 and march the same 5 time
    units at nu = nu_true, exactly the forward map the inversion differentiates

That framing matters. Warming each mesh up independently to t = 80 and comparing
at t = 85 would measure phase drift, not discretization: the meshes differ in
Strouhal number by ~0.2%, which over 80 convective times is most of a radian.
Starting both from one state limits the phase they can accumulate to the 5-unit
window, where a 0.2% rate difference is worth 0.01 time units.

u0 lives on the production mesh, so it is interpolated onto the fine mesh. That
interpolation is not discretely divergence-free on the fine mesh, and the error it
introduces is the same order as the quantity being measured, so it has to be
separated out rather than assumed small. Note that the difference of the t = 0 and
t = T error *norms* is not that separation -- they are norms of different error
fields, which may cancel or reinforce. The separation done here marches the
production solver a second time from the round-tripped initial condition, which
isolates the interpolation perturbation with no mesh difference in play, and then
projects the total error field onto it. The orthogonal remainder is the part
interpolation cannot explain.

Norm: the production-mesh P2 mass matrix, i.e. a true L2(Omega) norm rather than a
nodal average.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "cases", "case4_cylinder"))

from cylinder_solver import CylinderConfig, create_solver   # noqa: E402

NU = 0.01
T_WIN = 5.0
DT = 0.005
REFINE = 1.5           # same ratio as the OpenFOAM family


def p2_eval_operator(mesh, pts, k=30):
    """Sparse (n_pts, n_p2) operator evaluating a P2 scalar field at pts.

    Same algebra as cylinder_solver.build_probe_operator, but a KD-tree on element
    centroids restricts the barycentric search to the k nearest candidates. The
    original scans every element for every point, which is fine for 16 probes and
    hopeless for the ~25k nodes needed here.
    """
    coords, elem = mesh.coords_p2, mesh.elem_p2
    v = coords[elem[:, :3]]
    v0 = v[:, 0]
    d1 = v[:, 1] - v0
    d2 = v[:, 2] - v0
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    cent = v.mean(axis=1)
    tree = cKDTree(cent)
    k = min(k, len(cent))
    _, cand = tree.query(pts, k=k)

    rows, cols, data = [], [], []
    for j, (px, py) in enumerate(pts):
        c = cand[j]
        dx = px - v0[c, 0]
        dy = py - v0[c, 1]
        xi = (d2[c, 1] * dx - d2[c, 0] * dy) / det[c]
        eta = (-d1[c, 1] * dx + d1[c, 0] * dy) / det[c]
        l0, l1, l2 = 1.0 - xi - eta, xi, eta
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        # nearest containing element, else the least-bad one (curved boundary:
        # a fine-mesh node can sit just outside the coarse polygonal circle)
        i = int(np.argmax(inside)) if inside.any() else \
            int(np.argmax(np.minimum.reduce([l0, l1, l2])))
        e = int(c[i])
        a, b, cc = l0[i], l1[i], l2[i]
        phi = np.array([a * (2 * a - 1), b * (2 * b - 1), cc * (2 * cc - 1),
                        4 * a * b, 4 * b * cc, 4 * a * cc])
        for n in range(6):
            rows.append(j)
            cols.append(elem[e, n])
            data.append(phi[n])
    return csr_matrix((data, (rows, cols)), shape=(len(pts), mesh.n_p2))


def main():
    t_start = time.time()
    hc, hw, hf = 0.04, 0.08, 0.5                       # production
    fc, fw, ff = hc / REFINE, hw / REFINE, hf / REFINE  # finer reference

    print(f"production mesh: h_cyl={hc}, h_wake={hw}, h_far={hf}")
    mp, ap, sp = create_solver(CylinderConfig(h_cyl=hc, h_wake=hw, h_far=hf, dt=DT))
    print(f"  n_p2={mp.n_p2}  n_elem={mp.n_elem}  n_u={mp.n_u}", flush=True)

    print(f"fine mesh:       h_cyl={fc:.5f}, h_wake={fw:.5f}, h_far={ff:.5f}")
    mf, af, sf = create_solver(CylinderConfig(h_cyl=fc, h_wake=fw, h_far=ff, dt=DT))
    print(f"  n_p2={mf.n_p2}  n_elem={mf.n_elem}  n_u={mf.n_u}"
          f"   ({mf.n_p2/mp.n_p2:.2f}x the production nodes)", flush=True)

    z = np.load(os.path.join(REPO, "cylinder", "history", "saturated.npz"),
                allow_pickle=True)
    u0 = np.asarray(z["u0"])
    assert u0.size == mp.n_u, f"u0 size {u0.size} != production n_u {mp.n_u}"

    # production -> fine (to start the reference) and fine -> production (to
    # compare); both directions are needed and both are built once.
    print("building interpolation operators ...", flush=True)
    P_pf = p2_eval_operator(mp, mf.coords_p2)      # evaluate prod field at fine nodes
    P_fp = p2_eval_operator(mf, mp.coords_p2)      # evaluate fine field at prod nodes
    print(f"  done ({time.time()-t_start:.0f}s)", flush=True)

    u0f = np.concatenate([P_pf @ u0[:mp.n_p2], P_pf @ u0[mp.n_p2:]])

    # L2 norm on the production space
    M = ap.assemble_mass_p2()
    Mu = bmat([[M, None], [None, M]], format="csr")

    def rel_l2(a, b):
        d = a - b
        return float(np.sqrt((d @ (Mu @ d)) / (b @ (Mu @ b))))

    # t = 0: interpolation round trip, the floor of this measurement
    u0_rt = np.concatenate([P_fp @ u0f[:mf.n_p2], P_fp @ u0f[mf.n_p2:]])
    e0 = rel_l2(u0_rt, u0)
    print(f"\nt=0 interpolation round-trip error: {e0:.4e}"
          f"   <- floor of the comparison", flush=True)

    print(f"\nmarching both to T={T_WIN} at nu={NU}, dt={DT} "
          f"({int(T_WIN/DT)} steps) ...", flush=True)
    t0 = time.time()
    op = sp.solve_forward(T=T_WIN, nu=NU, u0=u0, dt=DT, ramp_T=0.0,
                          adaptive=False)
    tp_ = time.time() - t0
    print(f"  production done ({tp_:.0f}s)", flush=True)
    t0 = time.time()
    of = sf.solve_forward(T=T_WIN, nu=NU, u0=u0f, dt=DT, ramp_T=0.0,
                          adaptive=False)
    print(f"  fine done ({time.time()-t0:.0f}s)", flush=True)

    uT_p = op["u"] if "u" in op else op["u_final"]
    uT_f = of["u"] if "u" in of else of["u_final"]
    uT_f_on_p = np.concatenate([P_fp @ uT_f[:mf.n_p2], P_fp @ uT_f[mf.n_p2:]])

    # --- separating interpolation from discretization ------------------------
    # Subtracting the t=0 norm from the t=T norm is NOT a decomposition: these
    # are norms of different error fields, which may cancel or reinforce. To
    # separate them properly, march the PRODUCTION solver a second time from the
    # round-tripped initial condition. That isolates how the interpolation
    # perturbation alone propagates over the window, with no mesh difference in
    # play, giving an error field that can be projected out of the total.
    print("  production restart from the round-tripped u0 "
          "(isolates interpolation) ...", flush=True)
    t0 = time.time()
    orp = sp.solve_forward(T=T_WIN, nu=NU, u0=u0_rt, dt=DT, ramp_T=0.0,
                           adaptive=False)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
    uT_p_rt = orp["u"] if "u" in orp else orp["u_final"]

    eT = rel_l2(uT_p, uT_f_on_p)
    # component split, on the production space
    npd = mp.n_p2
    dU = uT_p[:npd] - uT_f_on_p[:npd]
    dV = uT_p[npd:] - uT_f_on_p[npd:]
    eU = float(np.sqrt((dU @ (M @ dU)) / (uT_f_on_p[:npd] @ (M @ uT_f_on_p[:npd]))))
    eV = float(np.sqrt((dV @ (M @ dV)) / (uT_f_on_p[npd:] @ (M @ uT_f_on_p[npd:]))))

    print("\n" + "=" * 66)
    print(f"eps_u^solver (Test 4, cylinder) = {eT:.3e}")
    print("=" * 66)
    print(f"  velocity field at T = {T_WIN}, relative L2 over the whole domain")
    print(f"  u_x component            : {eU:.3e}")
    print(f"  u_y component            : {eV:.3e}")
    print(f"  interpolation floor (t=0): {e0:.3e}")

    # proper split: project the total error field onto the interpolation-induced
    # error field; the orthogonal remainder is what interpolation cannot explain
    d_tot = uT_p - uT_f_on_p
    d_int = uT_p - uT_p_rt                     # interpolation only, same mesh
    nrm = lambda v: float(np.sqrt(v @ (Mu @ v)))
    ref = nrm(uT_f_on_p)
    ni = nrm(d_int)
    if ni > 0:
        cosang = float(d_tot @ (Mu @ d_int)) / (nrm(d_tot) * ni)
        along = float(d_tot @ (Mu @ d_int)) / ni          # signed component
        perp = np.sqrt(max(nrm(d_tot) ** 2 - along ** 2, 0.0))
    else:
        cosang = along = float("nan"); perp = nrm(d_tot)
    print(f"\n  decomposition of the total error field at T:")
    print(f"    interpolation propagated to T : {ni/ref:.3e}"
          f"   ({100*ni/nrm(d_tot):.1f}% of the total norm)")
    print(f"    alignment with the total      : cos = {cosang:+.3f}")
    print(f"    component along interpolation : {along/ref:.3e}")
    print(f"    component orthogonal to it    : {perp/ref:.3e}"
          f"   <- discretization, what interpolation cannot explain")
    print(f"\n  Richardson at order 2 with r = {REFINE}: the production-mesh error")
    print(f"  against an exact solution is ~{eT * REFINE**2 / (REFINE**2 - 1):.3e}")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "solver_floor.npz"),
             eps_u_solver=eT, eps_U=eU, eps_V=eV, eps_interp=e0,
             eps_interp_T=ni/ref, eps_perp=perp/ref, cos_align=cosang,
             n_p2_prod=mp.n_p2, n_p2_fine=mf.n_p2, T=T_WIN, dt=DT, nu=NU)
    print(f"\ntotal {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
