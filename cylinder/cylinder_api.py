"""High-level API for the cylinder viscosity-identification study.

One CylinderAPI holds a single CylinderRunConfig and exposes the whole pipeline
as timed methods that save Burgers-style history .npz files (runtime_sec +
cfg_snapshot + iteration arrays). Mirrors Burgers_1D/BurgersAPI.BurgersInverseAPI.

Stages
------
solve_forward()        FEM warmup -> saturated IC + one-period reference (prod mesh)
build_observation()    sparse wake-probe observation from the saturated IC
gradient_check()       discrete-adjoint dJ/dtheta vs central FD (cfg.adj_mesh)
invert_adjoint()       scalar nu inversion by BFGS over theta=log nu (cfg.adj_mesh)
train_pinn_forward()   forward PINN (cylinder_pinn_forward.train)
train_pinn_inverse()   inverse PINN (cylinder_pinn_inverse.train)

The forward solver (cylinder_solver.py) is used as-is; nothing in it is edited.
"""
import os
import time
import numpy as np
from scipy.optimize import minimize

from cylinder_solver import create_solver, CylinderAdjoint, build_probe_operator
from cylinder_config import CylinderRunConfig
import cylinder_pinn_forward
import cylinder_pinn_inverse


class CylinderAPI:
    def __init__(self, cfg: CylinderRunConfig = None):
        self.cfg = cfg if cfg is not None else CylinderRunConfig()
        os.makedirs(self.cfg.hist_dir, exist_ok=True)
        os.makedirs(self.cfg.fig_dir, exist_ok=True)

    # =====================================================================
    # 1. forward solve  ->  saturated IC + one-period reference (prod mesh)
    # =====================================================================
    def solve_forward(self, verbose=True):
        cfg = self.cfg
        mesh, asm, solver = create_solver(cfg.solver_cfg("prod"))
        if verbose:
            print(mesh.summary().split("\n")[0], "nu_true=%.6f" % cfg.nu_true, flush=True)

        t = time.time()
        warm = solver.solve_forward(T=cfg.warm_T, ramp_T=cfg.ramp_T, adaptive=True,
                                    cfl_target=cfg.cfl_target, blob_t=cfg.blob_t,
                                    blob_amp=cfg.blob_amp, save_dt=cfg.warm_save_dt,
                                    verbose=verbose)
        u0_sat = warm["u_final"]
        if verbose:
            print("warmup %.1fs |u0|max=%.3f" % (time.time()-t, np.abs(u0_sat).max()), flush=True)

        # one shedding-period reference trajectory from the saturated state
        period = solver.solve_forward(T=cfg.T_obs, nu=cfg.nu_true, u0=u0_sat, dt=cfg.dt,
                                      ramp_T=0.0, adaptive=False, save_dt=cfg.ref_save_dt,
                                      verbose=verbose)
        runtime = time.time() - t

        np.savez(cfg.saturated_path,
             u0=u0_sat,
             warm_times=warm["times"], warm_snaps=warm["snaps"],
             warm_CD=warm["CD"], warm_CL=warm["CL"], warm_t_force=warm["t_force"],
             times=period["times"], snaps=period["snaps"],
             period_times=period["times"], period_snaps=period["snaps"],
             t0a=0.0, nu_true=cfg.nu_true,
             CD=period["CD"], CL=period["CL"], t_force=period["t_force"],
             period_CD=period["CD"], period_CL=period["CL"], period_t_force=period["t_force"],
             runtime_sec=runtime, cfg_snapshot=str(cfg.snapshot()))
        np.savez(cfg.forward_path,
             warm_CD=warm["CD"], warm_CL=warm["CL"], warm_t_force=warm["t_force"],
             period_CD=period["CD"], period_CL=period["CL"], period_t_force=period["t_force"],
             t_force=period["t_force"], CD=period["CD"], CL=period["CL"],
             nu_true=cfg.nu_true, runtime_sec=runtime)
        if verbose:
            print("saved %s + %s (%.1fs)" % (cfg.saturated_path, cfg.forward_path, runtime), flush=True)

        return dict(mesh=mesh, solver=solver, u0=u0_sat, period=period, runtime_sec=runtime)

    # =====================================================================
    # 2. observation build  (sparse wake probes, prod mesh)
    # =====================================================================
    def build_observation(self, verbose=True):
        cfg = self.cfg
        mesh, asm, solver = create_solver(cfg.solver_cfg("prod"))
        probe_xy = cfg.probe_xy()
        Pop = build_probe_operator(mesh, probe_xy)
        obs_steps = cfg.obs_steps()
        obs_t = np.array([n*cfg.dt for n in obs_steps])

        with np.load(cfg.saturated_path, mmap_mode="r") as z:
            u0 = np.array(z["u0"]); t0a = float(z["t0a"])
            saved_times = np.array(z["period_times"] if "period_times" in z.files else z["times"])
            saved_snaps = np.array(z["period_snaps"] if "period_snaps" in z.files else z["snaps"])

        idx = np.array([int(np.argmin(np.abs(saved_times - tt))) for tt in obs_t], dtype=int)
        can_reuse = (
            saved_snaps.ndim == 2 and saved_snaps.shape[1] == mesh.n_u and
            len(saved_times) > 0 and np.allclose(saved_times[idx], obs_t, atol=1e-12, rtol=0.0)
        )

        if can_reuse:
            obs_data = np.array([Pop @ saved_snaps[i] for i in idx])
            runtime = 0.0
            if verbose:
                print("reusing saved period snapshots for observations", flush=True)
        else:
            if verbose:
                print("saved period snapshots incompatible with requested observation times; rebuilding", flush=True)

            t = time.time()
            out = solver.solve_forward(T=cfg.T_obs, nu=cfg.nu_true, u0=u0, dt=cfg.dt,
                                       ramp_T=0.0, adaptive=False, store_all=True,
                                       save_dt=cfg.T_obs, verbose=verbose)
            U = out["all_u"]
            obs_data = np.array([Pop @ U[n] for n in obs_steps])       # (n_obs, 2*NP)
            runtime = time.time() - t

        if cfg.obs_noise > 0:
            rng = np.random.default_rng(cfg.obs_seed)
            obs_data = obs_data + cfg.obs_noise*rng.standard_normal(obs_data.shape)

        np.savez(cfg.obs_path, probe_xy=probe_xy, obs_t=obs_t, obs_data=obs_data,
                 t0a=t0a, T=cfg.T_obs, dt=cfg.dt, nu_true=cfg.nu_true,
                 obs_steps=np.array(obs_steps), noise=cfg.obs_noise,
                 runtime_sec=runtime, cfg_snapshot=str(cfg.snapshot()))
        if verbose:
            print("saved %s: obs_data %s |d| in [%.3f, %.3f] (%.1fs)"
                  % (cfg.obs_path, obs_data.shape, obs_data.min(), obs_data.max(), runtime), flush=True)
        return dict(mesh=mesh, probe_xy=probe_xy, obs_t=obs_t, obs_data=obs_data,
                    u0=u0, runtime_sec=runtime)

    # ---- shared mesh + observation setup for the adjoint stages ------------
    # mesh level is cfg.adj_mesh ('prod' by default, 'coarse' only for fast checks)
    def _adjoint_obs(self, verbose=True):
        cfg = self.cfg
        mesh, asm, solver = create_solver(cfg.solver_cfg(cfg.adj_mesh))
        adj = CylinderAdjoint(solver)
        probe_xy = cfg.probe_xy()
        Pop = build_probe_operator(mesh, probe_xy)
        obs_steps = cfg.obs_steps()
        if verbose:
            print(mesh.summary().split("\n")[0], "nu_true=%.8f" % cfg.nu_true, flush=True)
            print("probe P %s  K=%d dt=%.3f T_obs=%.2f obs_steps=%s"
                  % (Pop.shape, cfg.K, cfg.dt, cfg.T_obs, obs_steps), flush=True)

        u0 = None
        if cfg.adj_mesh == "prod" and os.path.exists(cfg.saturated_path):
            with np.load(cfg.saturated_path, mmap_mode="r") as z_sat:
                saved_u0 = np.array(z_sat["u0"])
            if saved_u0.size == mesh.n_u:
                u0 = saved_u0
                if verbose:
                    print("reusing saved saturated state", flush=True)
            elif verbose:
                print("saved saturated state incompatible with current adjoint mesh; rebuilding", flush=True)

        if u0 is not None and os.path.exists(cfg.obs_path):
            with np.load(cfg.obs_path, mmap_mode="r") as z_obs:
                saved_probe_xy = np.array(z_obs["probe_xy"])
                saved_obs_steps = np.array(z_obs["obs_steps"], dtype=int)
                obs_data = np.array(z_obs["obs_data"])

            if np.allclose(saved_probe_xy, probe_xy) and np.array_equal(saved_obs_steps, obs_steps):
                if verbose:
                    print("reusing saved probe observations", flush=True)
                return dict(solver=solver, adj=adj, Pop=Pop, probe_xy=probe_xy,
                            u0=u0, obs_steps=obs_steps, obs_data=obs_data)
            if verbose:
                print("saved probe observations incompatible with current setup; rebuilding", flush=True)

        t = time.time()
        if u0 is None:
            warm = solver.solve_forward(T=cfg.warm_T, ramp_T=cfg.ramp_T, adaptive=True,
                                        cfl_target=cfg.cfl_target, blob_t=cfg.blob_t,
                                        blob_amp=cfg.blob_amp, save_dt=cfg.warm_T, verbose=verbose)
            u0 = warm["u_final"]
        obs = solver.solve_forward(T=cfg.T_obs, nu=cfg.nu_true, u0=u0, dt=cfg.dt, ramp_T=0.0,
                                   adaptive=False, store_all=True, save_dt=cfg.T_obs)
        obs_data = [Pop @ obs["all_u"][n] for n in obs_steps]
        if verbose:
            print("IC+obs ready (%.1fs): %d snapshots, dim=%d"
                  % (time.time()-t, len(obs_data), obs_data[0].size), flush=True)
        return dict(solver=solver, adj=adj, Pop=Pop, probe_xy=probe_xy,
                    u0=u0, obs_steps=obs_steps, obs_data=obs_data)

    # =====================================================================
    # 3. gradient check  (adjoint dJ/dtheta vs central FD, cfg.adj_mesh)
    # =====================================================================
    def gradient_check(self, verbose=True):
        cfg = self.cfg
        s = self._adjoint_obs(verbose=verbose)
        adj, u0, obs_steps, obs_data = s["adj"], s["u0"], s["obs_steps"], s["obs_data"]
        nu_true = cfg.nu_true; theta_true = np.log(nu_true)

        t = time.time()
        J0, g0, _ = adj.value_and_grad_obs(theta_true, u0, obs_steps, obs_data, s["Pop"], cfg.dt)
        if verbose:
            print("at theta_true: J=%.3e grad=%.3e (both ~0)" % (J0, g0), flush=True)

        rows = dict(fracs=[], theta=[], nu=[], J=[], g_adj=[], g_fd=[], rel=[])
        worst = 0.0
        for frac in cfg.gc_fracs:
            theta = np.log(nu_true*frac)
            J, g_adj, info = adj.value_and_grad_obs(theta, u0, obs_steps, obs_data, s["Pop"], cfg.dt)
            Jp, _, _ = adj.value_and_grad_obs(theta + cfg.gc_eps, u0, obs_steps, obs_data, s["Pop"], cfg.dt)
            Jm, _, _ = adj.value_and_grad_obs(theta - cfg.gc_eps, u0, obs_steps, obs_data, s["Pop"], cfg.dt)
            g_fd = (Jp - Jm)/(2*cfg.gc_eps)
            rel = abs(g_adj - g_fd)/(abs(g_fd) + 1e-30); worst = max(worst, rel)
            rows["fracs"].append(frac); rows["theta"].append(theta); rows["nu"].append(info["nu"])
            rows["J"].append(J); rows["g_adj"].append(g_adj); rows["g_fd"].append(g_fd); rows["rel"].append(rel)
            if verbose:
                print("  x%.2f  nu=%.6f  J=%.4e  adj=% .6e  fd=% .6e  rel=%.2e"
                      % (frac, info["nu"], J, g_adj, g_fd, rel), flush=True)
        runtime = time.time() - t
        if verbose:
            print("worst rel_err = %.2e -> %s (%.1fs)"
                  % (worst, "PASS" if worst < 1e-3 else "FAIL", runtime), flush=True)

        np.savez(cfg.grad_check_path,
                 fracs=np.array(rows["fracs"]), theta=np.array(rows["theta"]),
                 nu=np.array(rows["nu"]), J=np.array(rows["J"]),
                 g_adj=np.array(rows["g_adj"]), g_fd=np.array(rows["g_fd"]),
                 rel=np.array(rows["rel"]), worst=worst, J0=J0, g0=g0,
                 nu_true=nu_true, runtime_sec=runtime, cfg_snapshot=str(cfg.snapshot()))
        if verbose:
            print("saved %s" % cfg.grad_check_path, flush=True)
        return dict(worst=worst, runtime_sec=runtime, **{k: np.array(v) for k, v in rows.items()})

    # =====================================================================
    # 4. adjoint inversion  (scalar nu by BFGS over theta=log nu, cfg.adj_mesh)
    # =====================================================================
    def invert_adjoint(self, verbose=True, nu0=None, save_path=None):
        cfg = self.cfg
        s = self._adjoint_obs(verbose=verbose)
        adj, u0, obs_steps, obs_data = s["adj"], s["u0"], s["obs_steps"], s["obs_data"]
        nu_true = cfg.nu_true

        nu0 = float(nu_true * cfg.inv_nu0_factor if nu0 is None else nu0)
        save_path = cfg.inverse_path if save_path is None else save_path
        theta0 = np.log(nu0)
        if verbose:
            print("start nu0=%.8f (Re=%.1f) rel_err=%.3f"
                  % (nu0, cfg.U_inf*cfg.D/nu0, abs(nu0-nu_true)/nu_true), flush=True)

        hist = {"theta": [], "nu": [], "J": [], "g": [], "rel": []}
        neval = [0]
        def objective(x):
            theta = float(x[0])
            J, g, info = adj.value_and_grad_obs(theta, u0, obs_steps, obs_data, s["Pop"], cfg.dt)
            neval[0] += 1
            rel = abs(info["nu"] - nu_true)/nu_true
            hist["theta"].append(theta); hist["nu"].append(info["nu"])
            hist["J"].append(J); hist["g"].append(g); hist["rel"].append(rel)
            if verbose:
                print("  eval %2d: nu=%.8f J=%.4e dJ/dtheta=% .4e rel_nu_err=%.3e"
                      % (neval[0], info["nu"], J, g, rel), flush=True)
            return J, np.array([g])

        t = time.time()
        res = minimize(objective, np.array([theta0]), method='BFGS', jac=True,
                       options={'maxiter': cfg.inv_maxiter, 'gtol': cfg.inv_gtol,
                                'disp': False, 'method_bfgs': cfg.inv_method_bfgs})
        runtime = time.time() - t
        nu_rec = float(np.exp(res.x[0])); rel_rec = abs(nu_rec - nu_true)/nu_true
        if verbose:
            print("\n=== inversion finished (%.1fs) ===" % runtime, flush=True)
            print("  nu_true=%.8f  nu_rec=%.8f (Re_rec=%.4f)  rel=%.3e"
                  % (nu_true, nu_rec, cfg.U_inf*cfg.D/nu_rec, rel_rec), flush=True)

        np.savez(save_path, nu_true=nu_true, nu_rec=nu_rec, rel_rec=rel_rec,
                 theta=np.array(hist["theta"]), nu=np.array(hist["nu"]),
                 J=np.array(hist["J"]), g=np.array(hist["g"]), rel=np.array(hist["rel"]),
                 nit=res.nit, nfev=neval[0], probe_xy=s["probe_xy"],
                 obs_steps=np.array(obs_steps), dt=cfg.dt,
                 runtime_sec=runtime, nu0=nu0, cfg_snapshot=str(cfg.snapshot()))
        if verbose:
            print("saved %s" % save_path, flush=True)
        return dict(nu_rec=nu_rec, nu_true=nu_true, rel_rec=rel_rec, nit=res.nit,
                    nfev=neval[0], runtime_sec=runtime, nu0=nu0, save_path=save_path,
                    **{k: np.array(v) for k, v in hist.items()})

    # =====================================================================
    # 5/6. PINN forward / inverse  (thin pass-through to the train modules)
    # =====================================================================
    def train_pinn_forward(self):
        return cylinder_pinn_forward.train(self.cfg)

    def profile_pinn_forward(self):
        return cylinder_pinn_forward.train(self.cfg, profile_only=True)

    def train_pinn_inverse(self):
        return cylinder_pinn_inverse.train(self.cfg)
