"""Denser probe observation for the inverse PINN: a 7x7 grid of wake probes over
[1.5,6] x [-1.5,1.5] (49 probes) sampled at the same 10 snapshots over one
shedding period. Same FEM recipe as make_probe_obs.py (march the t=100 snapshot
at nu_true, dt=0.01). Saves history/probe_obs2.npz. CPU job."""
import time
import numpy as np
from cylinder_solver import CylinderConfig, create_solver, build_probe_operator

NU_TRUE = 1.0/60.0

cfg = CylinderConfig(h_cyl=0.04, h_wake=0.08, h_far=0.5)
mesh, asm, solver = create_solver(cfg)
nps = mesh.n_p2
print("production mesh: n_p2=%d  saddle DOF=%d" % (nps, 2*nps + mesh.n_p1), flush=True)

# ---- 7x7 wake probe grid over [1.5,6] x [-1.5,1.5] ----
PX = np.linspace(1.5, 6.0, 7)
PY = np.linspace(-1.5, 1.5, 7)
probe_xy = np.array([(x, y) for x in PX for y in PY])      # 49 points
Pop = build_probe_operator(mesh, probe_xy)
dt = 0.01
K = 600
obs_steps = list(range(60, K + 1, 60))                     # 10 snapshots
print("probe P: shape %s (%d probes x 2 comps)  K=%d dt=%.3f T_obs=%.2f"
      % (Pop.shape, len(probe_xy), K, dt, K*dt), flush=True)

# ---- IC = exact t=100 snapshot (same state the PINN uses) ----
with np.load('history/validate.npz', mmap_mode='r') as z:
    times = np.array(z['times'])
    i0 = int(np.argmin(np.abs(times - 100.0)))
    u0 = np.array(z['snaps'][i0]); t0a = float(times[i0])
print("IC: validate t=%.4f (idx %d)" % (t0a, i0), flush=True)

t = time.time()
out = solver.solve_forward(T=K*dt, nu=NU_TRUE, u0=u0, dt=dt, ramp_T=0.0,
                           adaptive=False, store_all=True, save_dt=K*dt)
U = out['all_u']
obs_data = np.array([Pop @ U[n] for n in obs_steps])       # (10, 98)
obs_t = np.array([n*dt for n in obs_steps])
print("solve %.1fs; obs_data shape %s  |d| range [%.3f, %.3f]"
      % (time.time()-t, obs_data.shape, obs_data.min(), obs_data.max()), flush=True)

np.savez('history/probe_obs2.npz', probe_xy=probe_xy, obs_t=obs_t, obs_data=obs_data,
         t0a=t0a, T=K*dt, dt=dt, nu_true=NU_TRUE, obs_steps=np.array(obs_steps))
print("saved history/probe_obs2.npz", flush=True)
