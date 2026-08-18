"""Adjoint viscosity inversion on the COARSE mesh with the 3x3 wake-probe,
10-snapshot observation. Recovers scalar theta=log(nu) by SSBroyden2 BFGS from
an off-truth start (Re=40, i.e. nu = 1.5 nu_true), using the gradient-verified
discrete adjoint (value_and_grad_obs). Noise-free baseline. Run via SLURM."""
import time
import numpy as np
from scipy.optimize import minimize
from cylinder_solver import (CylinderConfig, create_solver, CylinderAdjoint,
                             build_probe_operator)

# ---- coarse mesh + observation protocol (same as the gradient check) ----
cfg = CylinderConfig(h_cyl=0.06, h_wake=0.12, h_far=0.6)
mesh, asm, solver = create_solver(cfg)
adj = CylinderAdjoint(solver)
print(mesh.summary().split("\n")[0], "nu_true=%.8f" % cfg.nu, flush=True)

PX, PY = [1.5, 3.0, 4.5], [-0.75, 0.0, 0.75]
probe_xy = np.array([(x, y) for x in PX for y in PY])
Pop = build_probe_operator(mesh, probe_xy)

dt = 0.01
K = 600                                                # ~one shedding period
obs_steps = list(range(60, K + 1, 60))                 # 10 snapshots
nu_true = cfg.nu
theta_true = np.log(nu_true)
print("K=%d dt=%.3f T_obs=%.2f obs_steps=%s" % (K, dt, K*dt, obs_steps), flush=True)

# ---- developed snapshot as the (fixed, known) inverse-problem IC ----
t = time.time()
warm = solver.solve_forward(T=15.0, ramp_T=2.0, adaptive=True, cfl_target=0.9,
                            blob_t=8.0, blob_amp=0.1, save_dt=15.0)
u0 = warm["u_final"]
print("warmup %.1fs, |u0|max=%.3f" % (time.time()-t, np.abs(u0).max()), flush=True)

# ---- synthetic observation at true nu ----
obs = solver.solve_forward(T=K*dt, nu=nu_true, u0=u0, dt=dt, ramp_T=0.0,
                           adaptive=False, store_all=True, save_dt=K*dt)
obs_data = [Pop @ obs["all_u"][n] for n in obs_steps]
print("obs generated: %d snapshots, dim=%d" % (len(obs_data), obs_data[0].size), flush=True)

# ---- optimization over scalar theta = log(nu) ----
nu0 = nu_true * 1.5                                     # Re=40 start
theta0 = np.log(nu0)
print("\nstart: nu0=%.8f (Re=%.1f), theta0=%.6f  | rel err=%.3f"
      % (nu0, cfg.U_inf*cfg.D/nu0, theta0, abs(nu0-nu_true)/nu_true), flush=True)

hist = {"theta": [], "nu": [], "J": [], "g": [], "rel": []}
neval = [0]

def objective(x):
    theta = float(x[0])
    J, g, info = adj.value_and_grad_obs(theta, u0, obs_steps, obs_data, Pop, dt)
    neval[0] += 1
    rel = abs(info["nu"] - nu_true)/nu_true
    hist["theta"].append(theta); hist["nu"].append(info["nu"])
    hist["J"].append(J); hist["g"].append(g); hist["rel"].append(rel)
    print("  eval %2d: nu=%.8f  J=%.4e  dJ/dtheta=% .4e  rel_nu_err=%.3e"
          % (neval[0], info["nu"], J, g, rel), flush=True)
    return J, np.array([g])

t = time.time()
# NOTE: SSBroyden1/2 are undefined for a SCALAR control -- their update has a
# 1/(1-N) exponent that divides by zero at N=1. Use the classic full-Hessian
# BFGS update (the patched-scipy default), which for one parameter is exactly
# the safeguarded secant/Newton line search the methodology intends.
res = minimize(objective, np.array([theta0]), method='BFGS', jac=True,
               options={'maxiter': 50, 'gtol': 1e-9, 'disp': False,
                        'method_bfgs': 'BFGS'})
wall = time.time() - t

nu_rec = float(np.exp(res.x[0]))
rel_rec = abs(nu_rec - nu_true)/nu_true
print("\n=== inversion finished (%.1fs) ===" % wall, flush=True)
print("  message : %s" % res.message, flush=True)
print("  iters=%d  fun_evals=%d" % (res.nit, neval[0]), flush=True)
print("  nu_true = %.8f" % nu_true, flush=True)
print("  nu_rec  = %.8f   (Re_rec=%.4f)" % (nu_rec, cfg.U_inf*cfg.D/nu_rec), flush=True)
print("  |nu_rec - nu_true|/nu_true = %.3e" % rel_rec, flush=True)
print("  final J = %.4e  |grad| = %.4e" % (res.fun, abs(res.jac[0])), flush=True)

np.savez('history/inverse_coarse.npz',
         nu_true=nu_true, nu_rec=nu_rec, rel_rec=rel_rec,
         theta=np.array(hist["theta"]), nu=np.array(hist["nu"]),
         J=np.array(hist["J"]), g=np.array(hist["g"]), rel=np.array(hist["rel"]),
         nit=res.nit, nfev=neval[0], probe_xy=probe_xy, obs_steps=obs_steps, dt=dt)
print("saved history/inverse_coarse.npz", flush=True)
