"""Re-run the production Darcy inversions with the convergence histories kept.

Bundle E records only the final errors -- it reads the length of the PINN history
for the evaluation counters and then discards the trace -- so the five-seed
convergence histories that Figure 3 now needs do not exist on disk. This re-runs
the production cell (sigma = 1%, gamma = 1e-3) for the adjoint and the PINN at the
same five seeds, keeping the traces.

The seed varies the noise realization as well as the network initialization, which
is what it does in bundle E: the Darcy adjoint starts from xi_0 = 0 and is
otherwise deterministic, so without a fresh noise draw its five runs would be
identical. That is why the adjoint has a spread in Table 2 at all.

Everything else -- mesh, KL basis, observation layout, regularization, optimizer
settings -- comes from DarcyConfig unchanged, so the final errors here should
reproduce the bundle-E records for the same cell.

Both arms record one entry per objective call, which is not one per iteration:
the quasi-Newton line search evaluates several trial points per accepted step. So
each arm also stores the indices of its accepted iterates -- adj_s*_it here, and
pinn_s*_it_idx from the history DarcyPINN now keeps -- and the convergence figure
plots against those, not against the raw evaluation count.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cases", "case2_darcy"))
sys.path.insert(0, os.path.join(REPO, "sweeps"))

from common.seeding import set_seed                                  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
NOISE, GAMMA = 0.01, 1.0e-2          # the production cell of Table 2, at the
                                     # tuned gamma* selected in Table 4
OUT = os.path.join(REPO, "results", "darcy_history_data.npz")


def main():
    import bundle_E as E
    from cfg import DarcyConfig
    from scipy.optimize import minimize
    import torch

    d, rec = {}, []
    for seed in SEEDS:
        cfg = DarcyConfig(gamma=GAMMA, noise_level=NOISE, noise_seed=1000 + seed,
                          eki_seed=2000 + seed, seed=seed,
                          device="cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.seed)
        inv, RF, coefs_true, m_true, y_obs, u_clean, obs_points = E._setup(cfg, seed)
        Phi = E._kl_basis(inv, RF, cfg)
        r = Phi.shape[0]
        nrm_true = np.linalg.norm(m_true)

        # ---------------------------------------------------------- adjoint
        # The objective is called once per line-search trial, so the recorded
        # history is not an iterate sequence. The callback marks which records
        # are accepted iterates; the figure plots those, and the count matches
        # the iteration budget the text quotes.
        hist = {"J": [], "misfit": [], "reg": [], "eps": []}
        seen, iterates = [], [0]
        t0 = time.time()

        def obj(xi):
            m = Phi.T @ xi
            J, grad_m, misfit, reg = inv.objective_and_gradient(m, y_obs)
            hist["J"].append(float(J))
            hist["misfit"].append(float(misfit))
            hist["reg"].append(float(reg))
            hist["eps"].append(float(np.linalg.norm(m - m_true) / nrm_true))
            seen.append(np.asarray(xi, float).copy())
            return J, Phi @ grad_m

        def cb(xk):
            for i in range(len(seen) - 1, -1, -1):
                if np.array_equal(seen[i], xk):
                    iterates.append(i)
                    return
            iterates.append(len(seen) - 1)

        res = minimize(fun=obj, x0=np.zeros(r), jac=True, method="BFGS",
                       callback=cb,
                       options=dict(maxiter=cfg.max_iter, gtol=cfg.gtol, disp=False,
                                    method_bfgs=cfg.opt_method,
                                    hess_inv0=np.eye(r), initial_scale=False))
        # scipy exits on the gradient test before the last callback fires, so the
        # returned point is appended by hand (as in the cylinder histories).
        last = next((i for i in range(len(seen) - 1, -1, -1)
                     if np.array_equal(seen[i], res.x)), len(seen) - 1)
        if last not in iterates:
            iterates.append(last)
        t_adj = time.time() - t0
        m_adj = Phi.T @ res.x
        e_adj = float(np.linalg.norm(m_adj - m_true) / nrm_true)
        for k, v in hist.items():
            d[f"adj_s{seed}_{k}"] = np.asarray(v, float)
        d[f"adj_s{seed}_it"] = np.asarray(iterates, int)
        d[f"adj_s{seed}_t"] = t_adj
        print(f"seed {seed} adjoint: {int(res.nit)} iterations / {len(hist['J'])} "
              f"evals, eps_f={e_adj:.4e}, {t_adj:.1f}s", flush=True)

        # ------------------------------------------------------------- PINN
        t0 = time.time()
        from common.instrument import Counters
        counters = Counters()
        m_rec, C_mesh, gamma_pinn = E._run_pinn(inv, RF, coefs_true, obs_points,
                                                y_obs, cfg, counters)
        t_pinn = time.time() - t0
        e_pinn = float(np.linalg.norm(m_rec - m_true) / nrm_true)
        # _run_pinn builds its own DarcyPINN; recover the trace it left behind
        assert E._LAST_PINN is not None, "bundle_E._run_pinn did not stash the PINN"
        ph = E._LAST_PINN.history
        for k_src, k_dst in (("loss", "loss"), ("loss_pde", "pde"),
                             ("loss_bc", "bc"), ("loss_data", "data"),
                             ("loss_reg", "reg"), ("m_error", "eps"),
                             ("iteration", "it")):
            d[f"pinn_s{seed}_{k_dst}"] = np.asarray(ph[k_src], float)
        d[f"pinn_s{seed}_it_idx"] = np.asarray(ph["bfgs_iterates"], int)
        d[f"pinn_s{seed}_t"] = t_pinn
        print(f"seed {seed} PINN:    {len(ph['bfgs_iterates'])} quasi-Newton "
              f"iterates / {len(ph['loss'])} evals, eps_f={e_pinn:.4e}, "
              f"{t_pinn:.0f}s", flush=True)
        rec.append((seed, e_adj, t_adj, e_pinn, t_pinn))

    d["seeds"] = np.array(SEEDS)
    d["noise"], d["gamma"] = NOISE, GAMMA
    np.savez(OUT, **d)
    a = np.array([[x[1], x[2], x[3], x[4]] for x in rec])
    print(f"\nadjoint eps_f {a[:,0].mean():.3e} +- {a[:,0].std(ddof=1):.1e}  "
          f"t {a[:,1].mean():.1f}s   (Table 2: 2.87+-0.13e-1, 1.9s)")
    print(f"PINN    eps_f {a[:,2].mean():.3e} +- {a[:,2].std(ddof=1):.1e}  "
          f"t {a[:,3].mean():.0f}s   (Table 2: 2.70+-0.18e-1, 196s)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
