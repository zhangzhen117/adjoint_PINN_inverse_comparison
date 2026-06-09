"""Adjoint gradient verification for theta = log(nu) with the SPARSE PROBE
observation protocol: a 3x3 grid of wake probes sampled at 10 snapshots over
~one shedding period. Generates the synthetic observation d^k = P u^{n_k}(nu_true)
from a developed snapshot, then compares the discrete-adjoint dJ/dtheta against
central finite differences at several theta offsets. Run via SLURM (CPU-bound).

Pass: rel err < 1e-3 between adjoint and central-FD gradient at every offset.
"""
import time
import numpy as np
from cylinder_solver import (CylinderConfig, create_solver, CylinderAdjoint,
                             build_probe_operator)

# ---- coarse mesh -> fast gradient check (rate is mesh-independent) ----
cfg = CylinderConfig(h_cyl=0.06, h_wake=0.12, h_far=0.6)
mesh, asm, solver = create_solver(cfg)
adj = CylinderAdjoint(solver)
print(mesh.summary().split("\n")[0], "nu_true=%.6f" % cfg.nu, flush=True)

# ---- observation protocol: 3x3 wake probes, 10 snapshots over ~one period ----
PX = [1.5, 3.0, 4.5]
PY = [-0.75, 0.0, 0.75]
probe_xy = np.array([(x, y) for x in PX for y in PY])      # 9 points
Pop = build_probe_operator(mesh, probe_xy)
print("probe operator P: shape", Pop.shape, " (9 probes x 2 comps)", flush=True)

dt = 0.01
K = 600                                                    # ~one period (St~0.167 -> T~6.0)
obs_steps = list(range(60, K + 1, 60))                     # 10 snapshots: 60,120,...,600
print("K=%d dt=%.3f T_obs=%.2f  obs_steps=%s" % (K, dt, K*dt, obs_steps), flush=True)

nu_true = cfg.nu
theta_true = np.log(nu_true)

# ---- developed snapshot used as the inverse-problem IC ----
t = time.time()
warm = solver.solve_forward(T=15.0, ramp_T=2.0, adaptive=True, cfl_target=0.9,
                            blob_t=8.0, blob_amp=0.1, save_dt=15.0)
u0 = warm["u_final"]
print("warmup %.1fs, |u0|max=%.3f" % (time.time()-t, np.abs(u0).max()), flush=True)

# ---- synthetic observation at true nu: d^k = P u^{n_k} ----
t = time.time()
obs = solver.solve_forward(T=K*dt, nu=nu_true, u0=u0, dt=dt, ramp_T=0.0,
                           adaptive=False, store_all=True, save_dt=K*dt)
Utrue = obs["all_u"]
obs_data = [Pop @ Utrue[n] for n in obs_steps]
print("obs generated (%.1fs): %d snapshots, probe vec dim=%d"
      % (time.time()-t, len(obs_data), obs_data[0].size), flush=True)

# ---- sanity: at theta_true J~0 and grad~0 ----
J0, g0, info0 = adj.value_and_grad_obs(theta_true, u0, obs_steps, obs_data, Pop, dt)
print("at theta_true: J=%.3e  grad=%.3e  (both should be ~0)" % (J0, g0), flush=True)

# ---- gradient check at offsets ----
eps = 1e-6
print("\n  offset   theta        nu         J          adj_grad      fd_grad       rel_err",
      flush=True)
worst = 0.0
for frac in (0.7, 0.85, 1.2, 1.5):
    theta = np.log(nu_true*frac)
    J, g_adj, info = adj.value_and_grad_obs(theta, u0, obs_steps, obs_data, Pop, dt)
    Jp, _, _ = adj.value_and_grad_obs(theta + eps, u0, obs_steps, obs_data, Pop, dt)
    Jm, _, _ = adj.value_and_grad_obs(theta - eps, u0, obs_steps, obs_data, Pop, dt)
    g_fd = (Jp - Jm)/(2*eps)
    rel = abs(g_adj - g_fd)/(abs(g_fd) + 1e-30)
    worst = max(worst, rel)
    print("  x%.2f   % .4f   %.6f  %.4e  % .6e  % .6e  %.2e"
          % (frac, theta, info["nu"], J, g_adj, g_fd, rel), flush=True)

print("\nworst rel_err = %.2e  ->  %s" % (worst, "PASS" if worst < 1e-3 else "FAIL"),
      flush=True)
