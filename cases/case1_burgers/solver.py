"""
Forward solver for 1D Burgers with periodic BCs (Crank-Nicolson + Newton).

PDE:  u_t + u * u_x - nu * u_xx = s(x),   x in [0, L), periodic, t in [0, T].

Provides:
  - make_D_L          (periodic FD operators)
  - BurgersFDCore     (CN + Newton time stepper)
  - burgers_forward   (convenience wrapper)
  - initial_condition
  - true_force
  - rel_l2_error
  - test_convergence
"""

import numpy as np
from cfg import BurgersConfig


# ================================================================
# FD operators (periodic)
# ================================================================

def make_D_L(N: int, L: float):
    """Periodic centered-difference first-derivative D and Laplacian Lx."""
    dx = L / N
    D = np.zeros((N, N), dtype=float)
    Lx = np.zeros((N, N), dtype=float)
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        D[i, ip] += 0.5 / dx
        D[i, im] += -0.5 / dx
        Lx[i, i] += -2.0 / dx**2
        Lx[i, ip] += 1.0 / dx**2
        Lx[i, im] += 1.0 / dx**2
    return D, Lx


# ================================================================
# Crank-Nicolson core
# ================================================================

class BurgersFDCore:
    """Finite-difference core (periodic, Crank-Nicolson in time)."""

    def __init__(self, cfg: BurgersConfig):
        self.cfg = cfg
        self.N, self.L = cfg.N, cfg.L
        self.nu, self.T = cfg.nu, cfg.t_final
        self.x = np.linspace(0.0, self.L, self.N, endpoint=False)
        self.D, self.Lx = make_D_L(self.N, self.L)
        self.dx = self.L / self.N
        # snap dt to hit T exactly
        self.Nt = int(np.round(self.T / cfg.dt))
        self.dt = self.T / self.Nt
        # Newton settings
        self.newton_tol = 1e-10
        self.newton_maxit = 20

    def _rhs(self, u, s):
        """f(u, s) = -u * (D u) + nu * L u + s"""
        return -u * (self.D @ u) + self.nu * (self.Lx @ u) + s

    def _jacobian_rhs(self, u):
        """J_f(u) = -diag(D u) - diag(u) @ D + nu * L"""
        Du = self.D @ u
        return -np.diag(Du) - np.diag(u) @ self.D + self.nu * self.Lx

    def step(self, u_n, s):
        """One Crank-Nicolson step with Newton solver."""
        dt = self.dt
        f_n = self._rhs(u_n, s)
        u = u_n + dt * f_n  # explicit Euler predictor

        for k in range(self.newton_maxit):
            f_u = self._rhs(u, s)
            r = u - u_n - 0.5 * dt * (f_n + f_u)
            if np.linalg.norm(r, ord=np.inf) < self.newton_tol:
                break
            Jf = self._jacobian_rhs(u)
            J = np.eye(self.N) - 0.5 * dt * Jf
            du = np.linalg.solve(J, -r)
            u = u + du
            if np.linalg.norm(du, ord=np.inf) < 1e-14:
                break
        return u

    def forward(self, u0, s, store_trajectory=False):
        """Integrate from t=0 to t=T. Returns dict with 'u_final' (and 'U_hist')."""
        u = u0.copy()
        U_hist = None
        if store_trajectory:
            U_hist = np.zeros((self.Nt + 1, self.N), dtype=float)
            U_hist[0] = u
        for n in range(self.Nt):
            u = self.step(u, s)
            if store_trajectory:
                U_hist[n + 1] = u
        out = {"u_final": u}
        if store_trajectory:
            out["U_hist"] = U_hist
        return out


def burgers_forward(u0, force, cfg, store_trajectory=False):
    """Convenience wrapper for forward solve."""
    core = BurgersFDCore(cfg)
    return core.forward(u0, force, store_trajectory=store_trajectory)


# ================================================================
# Problem setup
# ================================================================

def initial_condition(x):
    """u0(x) = sin(x)"""
    return np.sin(x)


def true_force(x):
    """Ground-truth forcing s(x) = sin(2x)."""
    return np.sin(2 * x)


def rel_l2_error(u_pred, u_ref):
    """Relative L2 error."""
    return np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-30)


# ================================================================
# Convergence test
# ================================================================

def test_convergence(N_min=32, level=6, Cdt=1.0, t_final=1.0):
    """Spatial mesh convergence (dt scaled with dx)."""
    L = 2 * np.pi
    nu = 0.015
    Ns = [N_min * 2**i for i in range(level)]

    # Reference on finest grid
    Nfine = Ns[-1]
    dxfine = L / Nfine
    dtfine = Cdt * dxfine
    cfg_fine = BurgersConfig(N=Nfine, dt=dtfine, t_final=t_final, nu=nu)
    xfine = np.linspace(0, L, Nfine, endpoint=False)
    u0_fine = initial_condition(xfine)
    s_fine = true_force(xfine)
    out_fine = burgers_forward(u0_fine, s_fine, cfg_fine)
    u_ref_fine = out_fine["u_final"]

    results = []
    for N in Ns[:-1]:
        dx = L / N
        dt = Cdt * dx
        cfg = BurgersConfig(N=N, dt=dt, t_final=t_final, nu=nu)
        x = np.linspace(0, L, N, endpoint=False)
        u0 = initial_condition(x)
        s = true_force(x)
        out = burgers_forward(u0, s, cfg)
        u_num = out["u_final"]

        ratio = Nfine // N
        u_ref_on_coarse = u_ref_fine[::ratio]
        err = u_num - u_ref_on_coarse

        l2_ref = np.sqrt(np.mean(u_ref_on_coarse**2))
        rel_l2 = np.sqrt(np.mean(err**2)) / l2_ref if l2_ref > 0 else np.nan

        linf_ref = np.max(np.abs(u_ref_on_coarse))
        rel_linf = np.max(np.abs(err)) / linf_ref if linf_ref > 0 else np.nan

        results.append((N, dx, dt, rel_l2, rel_linf))

    print("N\t dx\t\t dt\t\t relL2\t\t relLinf")
    for N, dx, dt, rel_l2, rel_linf in results:
        print(f"{N}\t {dx:.5e}\t {dt:.5e}\t {rel_l2:.5e}\t {rel_linf:.5e}")

    print("\nEstimated order (from relative L2):")
    for k in range(len(results) - 1):
        N1, dx1, _, rel1, _ = results[k]
        N2, dx2, _, rel2, _ = results[k + 1]
        if rel2 > 0:
            p = np.log(rel1 / rel2) / np.log(dx1 / dx2)
            print(f"  N={N1} -> N={N2}: p ~ {p:.2f}")

    return results
