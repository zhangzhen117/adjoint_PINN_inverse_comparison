import os
import time
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
from numpy.linalg import LinAlgError, cholesky
from scipy.optimize import minimize

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from cfg import AllenCahn3DConfig

try:
    from common.seeding import set_seed
except ImportError:  # benchmark dir without the repo on the path
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.seeding import set_seed
from common.pinn_arch import AdaptiveActivation, FourierEmbedding
from solver import reaction_true, rel_l2_error

torch.set_default_dtype(torch.float64)


class MLP(nn.Module):
    """Plain MLP, optionally with a Fourier input embedding and/or an adaptive
    activation, for the architecture study of referees R1.5 and R3.4.

    ``act`` accepts "tanh", "gelu", "silu" or "adaptive" (layer-wise L-LAAF with a
    tanh base). ReLU is deliberately not offered: the Allen-Cahn residual needs
    d2u/dx2 and ReLU's second derivative vanishes almost everywhere, so the PDE
    loss would be identically zero rather than merely poor.

    ``fourier`` is ``None`` or ``{"embed_dim": m, "embed_scale": s}``, prepending
    [sin(Bx), cos(Bx)] with fixed B ~ N(0, s^2).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int,
                 act: str = "tanh", fourier: dict | None = None):
        super().__init__()
        self.embed = None
        d = in_dim
        if fourier:
            self.embed = FourierEmbedding(in_dim, **fourier)
            d = self.embed.out_dim

        name = act.lower()
        if name == "relu":
            raise ValueError("ReLU has a vanishing second derivative; the PDE "
                             "residual for Allen-Cahn requires d2u/dx2")
        acts = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}
        if name == "adaptive":
            make_act = lambda: AdaptiveActivation("tanh", n=10.0)
        else:
            cls = acts.get(name, nn.Tanh)
            make_act = cls

        dims = [d] + [hidden] * layers + [out_dim]
        modules = []
        for idx in range(len(dims) - 2):
            modules.append(nn.Linear(dims[idx], dims[idx + 1]))
            modules.append(make_act())
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*modules)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        return self.net(x)


def _sample_face_points(cfg: AllenCahn3DConfig, n: int, device: str, generator: torch.Generator):
    xyz = torch.rand((n, 3), generator=generator, device=device, dtype=torch.float64)
    xyz[:, 0] *= cfg.Lx
    xyz[:, 1] *= cfg.Ly
    xyz[:, 2] *= cfg.Lz
    t = torch.rand((n, 1), generator=generator, device=device, dtype=torch.float64) * cfg.t_final
    normals = torch.zeros((n, 3), device=device, dtype=torch.float64)
    faces = torch.randint(0, 6, (n,), generator=generator, device=device)
    for face in range(6):
        mask = faces == face
        if face == 0:
            xyz[mask, 0] = 0.0
            normals[mask, 0] = -1.0
        elif face == 1:
            xyz[mask, 0] = cfg.Lx
            normals[mask, 0] = 1.0
        elif face == 2:
            xyz[mask, 1] = 0.0
            normals[mask, 1] = -1.0
        elif face == 3:
            xyz[mask, 1] = cfg.Ly
            normals[mask, 1] = 1.0
        elif face == 4:
            xyz[mask, 2] = 0.0
            normals[mask, 2] = -1.0
        else:
            xyz[mask, 2] = cfg.Lz
            normals[mask, 2] = 1.0
    return xyz, t, normals


def _make_fixed_batch(cfg, device, xyz_ic_t, u0_t, xyz_T_t, uT_t, seed=12345):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    xyz_int = torch.rand((cfg.n_interior, 3), generator=gen, device=device, dtype=torch.float64)
    xyz_int[:, 0] *= cfg.Lx
    xyz_int[:, 1] *= cfg.Ly
    xyz_int[:, 2] *= cfg.Lz
    t_int = torch.rand((cfg.n_interior, 1), generator=gen, device=device, dtype=torch.float64) * cfg.t_final

    xyz_bc, t_bc, normals = _sample_face_points(cfg, cfg.n_bc_t, device, gen)

    n_ic_total = int(xyz_ic_t.shape[0])
    n_obs_total = int(xyz_T_t.shape[0])
    n_state_batch = int(getattr(cfg, "n_data_t", n_obs_total))

    if n_state_batch <= 0 or n_state_batch >= n_ic_total:
        ic_idx = torch.arange(n_ic_total, device=device)
    else:
        ic_idx = torch.randperm(n_ic_total, generator=gen, device=device)[:n_state_batch]

    if n_state_batch <= 0 or n_state_batch >= n_obs_total:
        obs_idx = torch.arange(n_obs_total, device=device)
    else:
        obs_idx = torch.randperm(n_obs_total, generator=gen, device=device)[:n_state_batch]

    xyz_ic_batch = xyz_ic_t.index_select(0, ic_idx)
    u0_batch = u0_t.index_select(0, ic_idx)

    xyz_T_batch = xyz_T_t.index_select(0, obs_idx)
    uT_batch = uT_t.index_select(0, obs_idx)

    return {
        "xyz_int": xyz_int,
        "t_int": t_int,
        "xyz_bc": xyz_bc,
        "t_bc": t_bc,
        "normals": normals,
        "xyz_ic": xyz_ic_batch,
        "t_ic": torch.zeros((xyz_ic_batch.shape[0], 1), device=device, dtype=torch.float64),
        "u_ic": u0_batch,
        "xyz_T": xyz_T_batch,
        "t_T": torch.full((xyz_T_batch.shape[0], 1), cfg.t_final, device=device, dtype=torch.float64),
        "u_T": uT_batch,
    }


def _gradients(u_fun, xyz, t):
    xyz = xyz.requires_grad_(True)
    t = t.requires_grad_(True)
    xyzt = torch.cat([xyz, t], dim=1)
    u = u_fun(xyzt)
    grad_xyz = torch.autograd.grad(u, xyz, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = grad_xyz[:, 0:1]
    u_y = grad_xyz[:, 1:2]
    u_z = grad_xyz[:, 2:3]
    u_xx = torch.autograd.grad(u_x, xyz, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyz, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, xyz, torch.ones_like(u_z), create_graph=True, retain_graph=True)[0][:, 2:3]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    return u, grad_xyz, u_xx, u_yy, u_zz, u_t


def _pinn_loss(cfg, U_theta, S_phi, batch):
    u, _, u_xx, u_yy, u_zz, u_t = _gradients(U_theta, batch["xyz_int"], batch["t_int"])
    su = S_phi(u)
    residual = u_t - cfg.eps**2 * (u_xx + u_yy + u_zz) - su
    l_res = torch.mean(residual**2)

    xyz_bc = batch["xyz_bc"].clone().requires_grad_(True)
    t_bc = batch["t_bc"].clone().requires_grad_(True)
    u_bc = U_theta(torch.cat([xyz_bc, t_bc], dim=1))
    grad_bc = torch.autograd.grad(u_bc, xyz_bc, torch.ones_like(u_bc), create_graph=True, retain_graph=True)[0]
    normal_derivative = torch.sum(batch["normals"] * grad_bc, dim=1, keepdim=True)
    l_bc = torch.mean(normal_derivative**2)

    u_ic_pred = U_theta(torch.cat([batch["xyz_ic"], batch["t_ic"]], dim=1))
    l_ic = torch.mean((u_ic_pred - batch["u_ic"])**2)

    u_T_pred = U_theta(torch.cat([batch["xyz_T"], batch["t_T"]], dim=1))
    l_data = torch.mean((u_T_pred - batch["u_T"])**2)

    reg = torch.zeros((), dtype=torch.float64, device=u.device)
    if cfg.alpha_reg > 0.0:
        for param in list(U_theta.parameters()) + list(S_phi.parameters()):
            reg = reg + torch.sum(param**2)
        reg = cfg.alpha_reg * reg

    total = cfg.w_res * l_res + cfg.w_bc * l_bc + cfg.w_ic * l_ic + cfg.w_dataT * l_data + reg
    return total, {
        "L_res": float(l_res.detach()),
        "L_bc": float(l_bc.detach()),
        "L_ic": float(l_ic.detach()),
        "L_data": float(l_data.detach()),
        "L_reg": float(reg.detach()),
    }


def _pack_params(models):
    params = []
    for model in models:
        params += list(model.parameters())
    return parameters_to_vector(params).detach()


def _unpack_params(theta_t, models):
    params = []
    for model in models:
        params += list(model.parameters())
    vector_to_parameters(theta_t, params)


def save_models(cfg, U_theta, S_phi, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"U_state_dict": U_theta.state_dict(), "S_state_dict": S_phi.state_dict(), "cfg": asdict(cfg)}, path)


def load_models(cfg, U_theta, S_phi, path):
    blob = torch.load(path, map_location=cfg.device)
    U_theta.load_state_dict(blob["U_state_dict"])
    S_phi.load_state_dict(blob["S_state_dict"])


def eval_S_error(S_phi, device):
    with torch.no_grad():
        u = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64, device=device).view(-1, 1)
        y_true = torch.tensor(reaction_true(u.cpu().numpy()), dtype=torch.float64, device=device).view(-1, 1)
        y_pred = S_phi(u)
        err = torch.linalg.norm(y_pred - y_true) / torch.linalg.norm(y_true)
    return float(err)


def build_models(cfg: AllenCahn3DConfig):
    device = cfg.device
    fe = ({"embed_dim": cfg.fourier_m_U, "embed_scale": cfg.fourier_scale_U}
          if getattr(cfg, "fourier_m_U", 0) else None)
    U_theta = MLP(4, 1, cfg.hidden_U, cfg.layers_U, cfg.act_U, fourier=fe).to(device).double()
    S_phi = MLP(1, 1, cfg.hidden_S, cfg.layers_S, cfg.act_S).to(device).double()
    return U_theta, S_phi


def train_pinn(u0: np.ndarray, uT_target: np.ndarray, cfg: AllenCahn3DConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = cfg.device
    adam_print_every = 100
    bfgs_print_every = 20
    x = np.linspace(0.0, cfg.Lx, cfg.Nx)
    y = np.linspace(0.0, cfg.Ly, cfg.Ny)
    z = np.linspace(0.0, cfg.Lz, cfg.Nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    xyz_flat = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])

    xyz_t = torch.tensor(xyz_flat, dtype=torch.float64, device=device)
    u0_t = torch.tensor(np.asarray(u0, dtype=float).reshape(-1, 1), dtype=torch.float64, device=device)
    uT_t = torch.tensor(np.asarray(uT_target, dtype=float).reshape(-1, 1), dtype=torch.float64, device=device)

    U_theta, S_phi = build_models(cfg)
    if cfg.is_load:
        load_models(cfg, U_theta, S_phi, cfg.pinn_scipy_model)

    # Collocation seed derived from the run seed, so different seeds get
    # different point clouds as well as different initializations.
    batch = _make_fixed_batch(cfg, device, xyz_t, u0_t, xyz_t, uT_t,
                              seed=12345 + int(cfg.seed))
    history = {
        "adam": {"loss": [], "L_res": [], "L_ic": [], "L_bc": [], "L_data": [], "rel_l2_uT": [], "rel_l2_f": [], "lr": []},
        "bfgs": {"loss": [], "grad_norm": [], "L_res": [], "L_ic": [], "L_bc": [], "L_data": [], "rel_l2_uT": [], "rel_l2_f": []},
    }

    def compute_rel_errors():
        with torch.no_grad():
            xyzt_T = torch.cat([xyz_t, torch.full((xyz_t.shape[0], 1), cfg.t_final, device=device, dtype=torch.float64)], dim=1)
            u_T_pred = U_theta(xyzt_T).detach().cpu().numpy().reshape(cfg.Nx, cfg.Ny, cfg.Nz)
        rel_uT = float(rel_l2_error(u_T_pred, uT_target))
        rel_f = eval_S_error(S_phi, device)
        return rel_uT, rel_f

    def print_progress(prefix: str, step: int, row: Dict[str, float], *, epoch: int = 0, lr: float | None = None):
        parts = [
            prefix,
            f"step={step:04d}",
            f"loss={row['loss']:.6e}",
        ]
        if epoch > 0:
            parts.append(f"epoch={epoch:02d}")
        if "grad_norm" in row:
            parts.append(f"|g|={row['grad_norm']:.3e}")
        parts.extend(
            [
                f"L_res={row['L_res']:.3e}",
                f"L_ic={row['L_ic']:.3e}",
                f"L_bc={row['L_bc']:.3e}",
                f"L_data={row['L_data']:.3e}",
                f"rel_uT={row['rel_l2_uT']:.3e}",
                f"rel_f={row['rel_l2_f']:.3e}",
            ]
        )
        if lr is not None:
            parts.append(f"lr={lr:.3e}")
        print(" ".join(parts))

    t0 = time.perf_counter()
    warmup_steps = max(0, int(cfg.scipy_pinn_adam_warmup_steps))
    if warmup_steps > 0:
        params = list(U_theta.parameters()) + list(S_phi.parameters())
        opt = torch.optim.Adam(params, lr=1.0e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, warmup_steps // 5), gamma=0.3)
        for _it in range(1, warmup_steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, loss_dict = _pinn_loss(cfg, U_theta, S_phi, batch)
            loss.backward()
            opt.step()
            scheduler.step()

            loss_val = float(loss.detach())
            history["adam"]["loss"].append(loss_val)
            for key in ["L_res", "L_ic", "L_bc", "L_data"]:
                history["adam"][key].append(loss_dict[key])
            lr_val = scheduler.get_last_lr()[0]
            history["adam"]["lr"].append(lr_val)
            rel_uT, rel_f = compute_rel_errors()
            history["adam"]["rel_l2_uT"].append(rel_uT)
            history["adam"]["rel_l2_f"].append(rel_f)
            if _it == 1 or _it % adam_print_every == 0 or _it == warmup_steps:
                print_progress(
                    "[PINN][adam]",
                    _it,
                    {
                        "loss": loss_val,
                        "L_res": loss_dict["L_res"],
                        "L_ic": loss_dict["L_ic"],
                        "L_bc": loss_dict["L_bc"],
                        "L_data": loss_dict["L_data"],
                        "rel_l2_uT": rel_uT,
                        "rel_l2_f": rel_f,
                    },
                    lr=lr_val,
                )

    with torch.no_grad():
        theta0 = _pack_params((U_theta, S_phi)).cpu().numpy()

    eval_hist = []
    iter_state = {"eval_start": 0, "bfgs_iter": 0}
    last_eval = {}

    def fun_and_jac(theta_np):
        theta_t = torch.tensor(theta_np, dtype=torch.float64, device=device)
        _unpack_params(theta_t, (U_theta, S_phi))

        for param in list(U_theta.parameters()) + list(S_phi.parameters()):
            if param.grad is not None:
                param.grad = None

        loss, loss_dict = _pinn_loss(cfg, U_theta, S_phi, batch)
        loss.backward()
        grads = [torch.zeros_like(param) if param.grad is None else param.grad for param in list(U_theta.parameters()) + list(S_phi.parameters())]
        gvec = parameters_to_vector(grads)

        loss_val = float(loss.detach())
        grad_norm = float(torch.linalg.norm(gvec).detach())
        rel_uT, rel_f = compute_rel_errors()
        row = {"loss": loss_val, "grad_norm": grad_norm, "rel_l2_uT": rel_uT, "rel_l2_f": rel_f, **loss_dict}
        eval_hist.append(row)
        last_eval.update(row)
        return loss_val, gvec.detach().cpu().numpy()

    def cb(_xk):
        start = iter_state["eval_start"]
        end = len(eval_hist)
        row = eval_hist[end - 1] if end > start else last_eval
        iter_state["bfgs_iter"] += 1
        for key in ["loss", "grad_norm", "L_res", "L_ic", "L_bc", "L_data", "rel_l2_uT", "rel_l2_f"]:
            history["bfgs"][key].append(row.get(key, 0.0))
        iter_state["eval_start"] = end
        if iter_state["bfgs_iter"] == 1 or iter_state["bfgs_iter"] % bfgs_print_every == 0:
            print_progress(
                "[PINN][bfgs]",
                iter_state["bfgs_iter"],
                row,
                epoch=bfgs_state["epoch"],
            )

    H0 = np.eye(len(theta0))
    bfgs_state = {"epoch": 0}
    # Plateau stop across outer epochs. A fixed epoch budget compares
    # configurations at an arbitrary point rather than at convergence: with
    # cfg.scipy_pinn_epochs = 15 the baseline is nearly flat but slower variants
    # are still improving by 10-20% in their final epochs, so any ranking would
    # measure the budget instead of the architecture. Set
    # cfg.pinn_epoch_plateau > 0 to stop once the loss stops improving.
    plateau_n = getattr(cfg, "pinn_epoch_plateau", 0)
    plateau_rtol = getattr(cfg, "pinn_epoch_rtol", 1.0e-4)
    epoch_best, epoch_best_i = float("inf"), 0
    history["stop_reason"] = "maxepochs"
    for epoch in range(1, cfg.scipy_pinn_epochs + 1):
        bfgs_state["epoch"] = epoch
        iter_state["bfgs_iter"] = 0
        res = minimize(
            fun=fun_and_jac,
            x0=theta0,
            jac=True,
            method="BFGS",
            tol=cfg.scipy_ftol,
            callback=cb,
            options=dict(
                maxiter=cfg.scipy_pinn_maxiter,
                gtol=cfg.scipy_gtol,
                disp=cfg.scipy_disp,
                method_bfgs=cfg.scipy_method_bfgs,
                hess_inv0=H0,
                initial_scale=False,
            ),
        )
        theta0 = res.x.copy()
        H_new = getattr(res, "hess_inv", None)
        if H_new is not None:
            H0 = 0.5 * (H_new + H_new.T)
            try:
                cholesky(H0)
            except LinAlgError:
                H0 = np.eye(len(theta0))
        else:
            H0 = np.eye(len(theta0))
        final_row = eval_hist[-1] if eval_hist else last_eval
        if final_row:
            print_progress(
                "[PINN][bfgs] done",
                iter_state["bfgs_iter"],
                final_row,
                epoch=epoch,
            )

        if plateau_n:
            cur = float(final_row.get("loss", float("nan"))) if final_row else float("nan")
            if cur == cur and cur < epoch_best * (1.0 - plateau_rtol):
                epoch_best, epoch_best_i = cur, epoch
            elif epoch - epoch_best_i >= plateau_n:
                history["stop_reason"] = "plateau"
                print(f"[PINN][bfgs] plateau: no improvement in {plateau_n} epochs, "
                      f"stopping at epoch {epoch}", flush=True)
                break

        batch = _make_fixed_batch(cfg, device, xyz_t, u0_t, xyz_t, uT_t,
                                  seed=epoch + 12345 + 1000 * int(cfg.seed))

    history["epochs_run"] = epoch
    t1 = time.perf_counter()
    history["runtime_sec"] = t1 - t0

    os.makedirs(os.path.dirname(cfg.pinn_scipy_path), exist_ok=True)
    for phase in ["adam", "bfgs"]:
        for key in history[phase]:
            history[phase][key] = np.asarray(history[phase][key], dtype=float)

    np.savez_compressed(
        cfg.pinn_scipy_path,
        **{f"adam_{key}": val for key, val in history["adam"].items()},
        **{f"bfgs_{key}": val for key, val in history["bfgs"].items()},
        runtime_sec=np.array(history["runtime_sec"]),
        cfg_snapshot=np.array(asdict(cfg), dtype=object),
    )
    save_models(cfg, U_theta, S_phi, cfg.pinn_scipy_model)
    return history


def reaction_from_model(S_phi: nn.Module, u_field: np.ndarray, device: str):
    with torch.no_grad():
        u_t = torch.tensor(np.asarray(u_field, dtype=float).reshape(-1, 1), dtype=torch.float64, device=device)
        f_t = S_phi(u_t)
    return f_t.detach().cpu().numpy().reshape(np.asarray(u_field).shape)
