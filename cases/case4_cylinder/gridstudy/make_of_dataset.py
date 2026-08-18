"""Build a PINN-ready dataset from the converged OpenFOAM L5 solution.

The inverse PINN reads two files: an initial condition (cfg.saturated_path) and a
probe observation set (cfg.obs_path). Both must come from the SAME trajectory --
feeding it the FEM initial state with OpenFOAM observations would pose a problem
with no solution, since the two wakes are at different phases. So both are taken
from the L5 run over the window [80, 85], which is the window whose fields were
written to disk.

Boundary nodes are not probed. 158 of the FEM P2 nodes lie exactly on the cylinder
and 25 on the inlet; a probe there returns a wall-adjacent cell value rather than
the prescribed one, so the Dirichlet values are imposed exactly, matching how the
FEM initial state was built. Interior nodes on the outer walls are nudged 1e-3
inside before probing and then have v set to zero.

Run stages (this file is called twice by run_of_dataset.slurm):
  --stage points   write system/icProbes with the nudged node coordinates
  --stage assemble read the probe output back and write the two npz files
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CASE = os.path.join(HERE, "cases", "L5_trans")
T0, T1 = 80.0, 85.0             # window start / end, both written to disk
OBS_DT = 0.5                    # paper's observation spacing
EPS = 1.0e-3                    # nudge for boundary nodes


def load_nodes():
    z = np.load(os.path.join(HERE, "fem_nodes.npz"))
    return (z["xy"], int(z["n_p2"]), z["bnd_inlet"], z["bnd_cyl"],
            z["bnd_walls"], z["bnd_outlet"])


def nudged(xy, inlet, cyl, walls, outlet):
    """Move boundary nodes just inside the fluid so a probe lands in a real cell."""
    p = xy.copy()
    r = np.linalg.norm(p[cyl], axis=1, keepdims=True)
    p[cyl] = p[cyl] / r * (0.5 + 2 * EPS)          # radially outward, into the fluid
    p[inlet, 0] += EPS
    p[outlet, 0] -= EPS
    p[walls, 1] -= np.sign(p[walls, 1]) * EPS
    return p


def stage_points():
    xy, n_p2, inlet, cyl, walls, outlet = load_nodes()
    p = nudged(xy, inlet, cyl, walls, outlet)
    lines = "\n".join(f"        ({x:.10g} {y:.10g} 0)" for x, y in p)
    txt = f"""icProbes
{{
    type            probes;
    libs            (sampling);
    fields          (U);
    interpolationScheme cellPoint;
    probeLocations
    (
{lines}
    );
}}
"""
    out = os.path.join(CASE, "system", "icProbes")
    open(out, "w").write(txt)
    print(f"wrote {out} with {len(p)} probe locations")


def read_probe_file(path, npts):
    """-> {time: (npts, 3)} from an OpenFOAM probes output file."""
    out = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        vecs = re.findall(r"\(([^)]*)\)", line)
        if len(vecs) != npts:
            continue
        t = float(line.split()[0])
        out[t] = np.array([[float(v) for v in s.split()] for s in vecs])
    return out


def stage_assemble():
    xy, n_p2, inlet, cyl, walls, outlet = load_nodes()

    # ---- IC and terminal field at the FEM nodes ----
    hits = []
    for d in os.listdir(os.path.join(CASE, "postProcessing", "icProbes")):
        f = os.path.join(CASE, "postProcessing", "icProbes", d, "U")
        if os.path.exists(f):
            hits.append(f)
    fields = {}
    for f in hits:
        fields.update(read_probe_file(f, n_p2))
    have = sorted(fields)
    print(f"icProbes times available: {have}")

    def pick(t):
        k = min(have, key=lambda s: abs(s - t))
        assert abs(k - t) < 1e-6, f"no field at t={t} (closest {k})"
        return fields[k]

    def to_fem_vector(v3):
        """(n_p2,3) OpenFOAM samples -> the FEM's [ux; uy] layout, BCs imposed."""
        u = v3[:, 0].copy()
        v = v3[:, 1].copy()
        u[cyl] = 0.0
        v[cyl] = 0.0            # no slip
        u[inlet] = 1.0
        v[inlet] = 0.0          # prescribed inflow
        v[walls] = 0.0          # u_y = 0, u_x free
        return np.concatenate([u, v])

    u0 = to_fem_vector(pick(T0))
    uT = to_fem_vector(pick(T1))

    # ---- 16-probe observations over the window ----
    from analyze import read_probes
    tp, up = read_probes(CASE)
    obs_t = np.arange(OBS_DT, T1 - T0 + 1e-9, OBS_DT)
    P = up.shape[1]
    obs = np.empty((len(obs_t), 2 * P))
    for j, d in enumerate(obs_t):
        for q in range(P):
            obs[j, q] = np.interp(T0 + d, tp, up[:, q, 0])
            obs[j, P + q] = np.interp(T0 + d, tp, up[:, q, 1])

    probe_xy = np.array([(x, y) for x in np.linspace(1, 3, 4)
                         for y in np.linspace(-1, 1, 4)])

    hist = os.path.join(REPO, "cases", "case4_cylinder", "history")
    np.savez(os.path.join(hist, "saturated_of.npz"),
             u0=u0, times=np.array([0.0, T1 - T0]),
             snaps=np.stack([u0, uT]), t0a=0.0, nu_true=0.01,
             source=f"OpenFOAM L5 (195432 cells) window [{T0},{T1}]")
    np.savez(os.path.join(hist, "probe_obs_of.npz"),
             probe_xy=probe_xy, obs_t=obs_t, obs_data=obs,
             t0a=0.0, T=T1 - T0, dt=0.005, nu_true=0.01, noise=0.0,
             obs_steps=np.arange(100, 1001, 100),
             source=f"OpenFOAM L5 (195432 cells) window [{T0},{T1}]")

    # sanity against the FEM data the paper used
    fem = np.load(os.path.join(hist, "probe_obs.npz"), allow_pickle=True)
    print(f"\nwrote saturated_of.npz  u0 |u|max={np.abs(u0).max():.4f}  "
          f"n={u0.size}")
    print(f"wrote probe_obs_of.npz  obs {obs.shape}  "
          f"range [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"  FEM obs range for comparison: "
          f"[{fem['obs_data'].min():.4f}, {fem['obs_data'].max():.4f}]")
    print(f"  ||obs_OF|| = {np.linalg.norm(obs):.4f}   "
          f"||obs_FEM|| = {np.linalg.norm(fem['obs_data']):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("points", "assemble"))
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    (stage_points if a.stage == "points" else stage_assemble)()
