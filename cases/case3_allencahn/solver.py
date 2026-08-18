import numpy as np
from scipy.fft import dctn, idctn
from scipy.sparse import eye, kron

from cfg import AllenCahn3DConfig


def laplacian_neumann_1d(n: int, h: float):
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    lap = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    lap[0, 1] = 2.0
    lap[-1, -2] = 2.0
    lap /= h**2
    return lap


def build_laplacian_neumann_3d(cfg: AllenCahn3DConfig):
    hx = cfg.Lx / (cfg.Nx - 1)
    hy = cfg.Ly / (cfg.Ny - 1)
    hz = cfg.Lz / (cfg.Nz - 1)
    lx = laplacian_neumann_1d(cfg.Nx, hx)
    ly = laplacian_neumann_1d(cfg.Ny, hy)
    lz = laplacian_neumann_1d(cfg.Nz, hz)

    ix = eye(cfg.Nx, format="csc")
    iy = eye(cfg.Ny, format="csc")
    iz = eye(cfg.Nz, format="csc")
    lap = kron(kron(iz, iy, format="csc"), lx, format="csc")
    lap = lap + kron(kron(iz, ly, format="csc"), ix, format="csc")
    lap = lap + kron(kron(lz, iy, format="csc"), ix, format="csc")
    return lap, hx, hy, hz


def make_xyz_grid(cfg: AllenCahn3DConfig):
    x = np.linspace(0.0, cfg.Lx, cfg.Nx)
    y = np.linspace(0.0, cfg.Ly, cfg.Ny)
    z = np.linspace(0.0, cfg.Lz, cfg.Nz)
    return np.meshgrid(x, y, z, indexing="ij")


def reaction_true(u):
    return u - u**3


def initial_condition(x, y, z):
    return np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)


def rel_l2_error(u_pred, u_ref):
    denom = np.linalg.norm(u_ref.ravel())
    if denom == 0.0:
        return np.linalg.norm(u_pred.ravel() - u_ref.ravel())
    return np.linalg.norm(u_pred.ravel() - u_ref.ravel()) / denom


def _reshape_field(u_flat, cfg: AllenCahn3DConfig):
    return np.asarray(u_flat, dtype=float).reshape(cfg.Nx, cfg.Ny, cfg.Nz)


def _flatten_field(u):
    return np.asarray(u, dtype=float).reshape(-1)


def _neumann_eigenvalues_1d(n: int, h: float):
    modes = np.arange(n, dtype=float)
    return -4.0 * np.sin(0.5 * np.pi * modes / max(n - 1, 1)) ** 2 / (h**2)


def make_solver_cache(cfg: AllenCahn3DConfig):
    hx = cfg.Lx / (cfg.Nx - 1)
    hy = cfg.Ly / (cfg.Ny - 1)
    hz = cfg.Lz / (cfg.Nz - 1)
    lam_x = _neumann_eigenvalues_1d(cfg.Nx, hx)
    lam_y = _neumann_eigenvalues_1d(cfg.Ny, hy)
    lam_z = _neumann_eigenvalues_1d(cfg.Nz, hz)
    lambda_sum = (
        lam_x[:, None, None]
        + lam_y[None, :, None]
        + lam_z[None, None, :]
    )
    return {"hx": hx, "hy": hy, "hz": hz, "lambda_sum": lambda_sum, "factored": {}}


def _get_linear_solver(
    cache,
    cfg: AllenCahn3DConfig,
    dt: float,
    mass_coeff: float = 1.0,
    diff_coeff: float = 1.0,
):
    key = (round(float(dt), 14), round(float(mass_coeff), 14), round(float(diff_coeff), 14))
    if key not in cache["factored"]:
        denom = mass_coeff - diff_coeff * dt * (cfg.eps**2) * cache["lambda_sum"]
        if np.any(np.abs(denom) < 1.0e-30):
            raise ZeroDivisionError("Encountered a zero spectral denominator in the DCT solver.")

        def solve(rhs_flat):
            rhs = np.asarray(rhs_flat, dtype=float).reshape(cfg.Nx, cfg.Ny, cfg.Nz)
            rhs_hat = dctn(rhs, type=1)
            sol_hat = rhs_hat / denom
            sol = idctn(sol_hat, type=1)
            return np.asarray(sol, dtype=float).reshape(-1)

        cache["factored"][key] = solve
    return cache["factored"][key]


def solve_allen_cahn_imex(cfg: AllenCahn3DConfig, u0, reaction_fun=None):
    reaction = reaction_true if reaction_fun is None else reaction_fun
    cache = make_solver_cache(cfg)

    t = 0.0
    dt_full = float(cfg.dt)
    tol_dt = 1.0e-14 * max(1.0, abs(dt_full))

    u_curr = _flatten_field(u0)
    f_curr = _flatten_field(reaction(_reshape_field(u_curr, cfg)))
    u_prev = None
    f_prev = None
    prev_dt = None

    u_hist = [u_curr.reshape(cfg.Nx, cfg.Ny, cfg.Nz).copy()]
    t_hist = [t]

    while t < cfg.t_final - 1.0e-14:
        dt = min(cfg.dt, cfg.t_final - t)
        use_bdf2 = (
            u_prev is not None
            and prev_dt is not None
            and abs(dt - prev_dt) <= tol_dt
            and abs(dt - dt_full) <= tol_dt
        )

        if use_bdf2:
            solve_linear = _get_linear_solver(cache, cfg, dt, mass_coeff=3.0, diff_coeff=2.0)
            rhs = 4.0 * u_curr - u_prev + 2.0 * dt * (2.0 * f_curr - f_prev)
        else:
            solve_linear = _get_linear_solver(cache, cfg, dt)
            rhs = u_curr + dt * f_curr

        u_new = np.asarray(solve_linear(rhs), dtype=float).reshape(-1)
        t += dt
        u_hist.append(_reshape_field(u_new, cfg).copy())
        t_hist.append(t)

        u_prev = u_curr
        f_prev = f_curr
        prev_dt = dt
        u_curr = u_new
        if t < cfg.t_final - 1.0e-14:
            f_curr = _flatten_field(reaction(_reshape_field(u_curr, cfg)))

    return np.stack(u_hist, axis=0), np.array(t_hist), cache


def make_cfg(nx: int, ny: int, nz: int, dt: float, t_final: float, eps: float = 1.0e-2):
    return AllenCahn3DConfig(Nx=nx, Ny=ny, Nz=nz, dt=dt, t_final=t_final, eps=eps)


def restrict_to_coarse(u_fine, ratio_x: int, ratio_y: int, ratio_z: int):
    return u_fine[::ratio_x, ::ratio_y, ::ratio_z]


def run_mesh_convergence(levels=3, n_min=9, c_dt=0.1, t_final=0.1, eps=1.0e-2):
    sizes = [(n_min - 1) * (2**k) + 1 for k in range(levels)]
    n_fine = sizes[-1]
    h_fine = 1.0 / (n_fine - 1)
    dt_fine = c_dt * h_fine**2
    cfg_fine = make_cfg(n_fine, n_fine, n_fine, dt_fine, t_final, eps)

    xx_fine, yy_fine, zz_fine = make_xyz_grid(cfg_fine)
    u0_fine = initial_condition(xx_fine, yy_fine, zz_fine)
    u_hist_fine, _, _ = solve_allen_cahn_imex(cfg_fine, u0_fine)
    u_ref = u_hist_fine[-1]

    results = []
    for n in sizes[:-1]:
        h = 1.0 / (n - 1)
        dt = c_dt * h**2
        cfg = make_cfg(n, n, n, dt, t_final, eps)
        xx, yy, zz = make_xyz_grid(cfg)
        u0 = initial_condition(xx, yy, zz)
        u_hist, _, _ = solve_allen_cahn_imex(cfg, u0)
        u_num = u_hist[-1]

        ratio = (n_fine - 1) // (n - 1)
        u_ref_coarse = restrict_to_coarse(u_ref, ratio, ratio, ratio)
        err = u_num - u_ref_coarse

        rel_l2 = np.linalg.norm(err.ravel()) / max(np.linalg.norm(u_ref_coarse.ravel()), 1.0e-30)
        rel_linf = np.max(np.abs(err)) / max(np.max(np.abs(u_ref_coarse)), 1.0e-30)
        results.append((n, h, dt, rel_l2, rel_linf))

    return results


def estimate_orders(results):
    orders = []
    for idx in range(len(results) - 1):
        n1, h1, _, e1, _ = results[idx]
        n2, h2, _, e2, _ = results[idx + 1]
        if e1 > 0.0 and e2 > 0.0:
            orders.append((n1, n2, np.log(e1 / e2) / np.log(h1 / h2)))
    return orders