"""Re-run the three Darcy estimators at gamma* and keep the recovered fields.

Figures 4 and 5 date from the submission and were produced at gamma = 1e-3, while
Table 2 and Figure 3 now report sigma = 1% at the tuned gamma* = 1e-2. Bundle E
keeps only the scalar errors, so the fields those figures need do not exist on
disk at the new weight and have to be recomputed.

Seed 0 only: both figures show a single reconstruction, as their captions say, and
the point of them is the spatial structure rather than the spread.

Everything except gamma comes from DarcyConfig unchanged, so the errors printed at
the end should match the bundle-E records for the same cell -- if they do not, the
fields are not the ones behind the table and should not be plotted.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "Darcy_New"))
sys.path.insert(0, os.path.join(REPO, "sweeps"))

from common.seeding import set_seed                                  # noqa: E402

SEED = 0
NOISE, GAMMA = 0.01, 1.0e-2
OUT = os.path.join(REPO, "results", "darcy_field_data.npz")


def main():
    import bundle_E as E
    from cfg import DarcyConfig
    from common.instrument import Counters
    import torch

    cfg = DarcyConfig(gamma=GAMMA, noise_level=NOISE, noise_seed=1000 + SEED,
                      eki_seed=2000 + SEED, seed=SEED,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    inv, RF, coefs_true, m_true, y_obs, u_clean, obs_points = E._setup(cfg, SEED)
    Phi = E._kl_basis(inv, RF, cfg)
    sigma_obs = cfg.noise_level * np.max(np.abs(u_clean))

    fields = {}
    for name, fn in (("adjoint", lambda c: E._run_adjoint(inv, Phi, y_obs, m_true, cfg, c)),
                     ("eki",     lambda c: E._run_eki(inv, Phi, y_obs, cfg, c, sigma_obs)),
                     ("pinn",    lambda c: E._run_pinn(inv, RF, coefs_true, obs_points,
                                                       y_obs, cfg, c)[0])):
        m = fn(Counters())
        u = inv.solve_forward(m)
        ef = float(np.linalg.norm(m - m_true) / np.linalg.norm(m_true))
        eu = float(np.linalg.norm(u - u_clean) / np.linalg.norm(u_clean))
        fields[f"m_{name}"] = m
        fields[f"u_{name}"] = u
        fields[f"eps_f_{name}"] = ef
        fields[f"eps_u_{name}"] = eu
        print(f"  {name:8s} eps_f {ef:.4f}  eps_u {eu:.4e}", flush=True)

    # m is piecewise constant on triangles, u is nodal, so both geometries are
    # needed to contour them the way the submitted figures did
    np.savez(OUT, m_true=m_true, u_true=u_clean,
             centroids=inv.element_centroids, nodes=inv.nodes,
             elements=inv.elements, obs_points=obs_points,
             noise=NOISE, gamma=GAMMA, seed=SEED, **fields)
    print(f"\nwrote {OUT}")
    print("  Table 2 at this cell: adjoint 0.2874, PINN 0.2697, EKI 0.2582")


if __name__ == "__main__":
    main()
