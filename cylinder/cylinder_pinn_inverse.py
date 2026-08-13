"""Inverse PINN for the cylinder flow: identify the scalar viscosity nu from
sparse wake-probe observations, with nu >= 0 enforced via theta = log(nu).

Canonical successor of cylinder_pinn_inv4.py, refactored into a single train(cfg)
entry point. Reads every flag from a CylinderRunConfig: net 32x3, NO
near-cylinder oversampling, the probe observation file (cfg.obs_path) and the FEM
IC (cfg.saturated_path), cold-start nu0=cfg.nu0_pinn, Adam -> SSBroyden2. Records
loss components + nu trajectory, saves cfg.pinn_inv_npz + cfg.pinn_inv_model + a
training-curve figure. Run in env opt_2nd (GPU). See methodology_Cylinder.md.
"""
import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from cylinder_solver import create_solver

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from common.pinn_arch import ModifiedMlp, PlainMlp

torch.set_default_dtype(torch.float64)


def train(cfg):
    DEV = cfg.device
    torch.manual_seed(cfg.seed_inv); np.random.seed(cfg.seed_inv)
    print("device =", DEV, flush=True)

    X0, X1, Y0, Y1 = cfg.x0, cfg.x1, cfg.y0, cfg.y1
    R = cfg.R
    NU_TRUE = cfg.nu_true
    NU0 = cfg.nu0_pinn
    MARGIN, R_NEAR, NEAR_FRAC = cfg.margin, cfg.r_near, cfg.near_frac
    HIDDEN, LAYERS = cfg.hidden, cfg.layers

    # ----------------------------------------------- IC + probe observation
    cyl_cfg = cfg.solver_cfg("prod")
    mesh, _, _ = create_solver(cyl_cfg)
    nps = mesh.n_p2; XY = mesh.coords_p2
    with np.load(cfg.saturated_path, mmap_mode='r') as z:
        times = np.array(z['times']); snaps = z['snaps']
        t0a = float(z['t0a']); snap0 = np.array(z['u0'])

    obs = np.load(cfg.obs_path)
    probe_xy = obs['probe_xy']; obs_t = obs['obs_t']; obs_data = obs['obs_data']
    T = float(obs['T'])
    assert abs(t0a - float(obs['t0a'])) < 1e-6, "IC/obs t0a mismatch"
    print("IC t=%.3f  T=%.4f (n_p2=%d)  obs: %d snaps x %d probes = %d scalars"
          % (t0a, T, nps, obs_data.shape[0], probe_xy.shape[0], obs_data.size), flush=True)

    IC_XY = XY.copy()
    IC_UV = np.column_stack([snap0[:nps], snap0[nps:2*nps]])
    IC_T = np.zeros(len(IC_XY))
    print("IC = all %d mesh nodes (exact)" % len(IC_XY), flush=True)

    def interp_pair(snap):
        uv = np.column_stack([snap[:nps], snap[nps:2*nps]])
        return LinearNDInterpolator(XY, uv), NearestNDInterpolator(XY, uv)
    i1 = int(np.argmin(np.abs(times - (t0a + T))))
    rf_lin, rf_nea = interp_pair(np.array(snaps[i1]))
    def eval_uv(lin, nea, pts):
        v = lin(pts); bad = np.isnan(v[:, 0])
        if bad.any(): v[bad] = nea(pts[bad])
        return v

    NP = probe_xy.shape[0]
    OBS_X = np.tile(probe_xy[:, 0], len(obs_t))
    OBS_Y = np.tile(probe_xy[:, 1], len(obs_t))
    OBS_TT = np.repeat(obs_t, NP)
    OBS_U = obs_data[:, :NP].ravel()
    OBS_V = obs_data[:, NP:].ravel()

    # ----------------------------------------------------------------- sampling
    def sample_fluid(n):
        out = np.empty((0, 2))
        while len(out) < n:
            c = np.random.rand(2*n, 2)
            c[:, 0] = X0 + c[:, 0]*(X1 - X0); c[:, 1] = Y0 + c[:, 1]*(Y1 - Y0)
            c = c[np.hypot(c[:, 0], c[:, 1]) > R + MARGIN]
            out = np.vstack([out, c])
        return out[:n]

    def sample_near(n):
        rr = np.sqrt(R**2 + np.random.rand(n)*(R_NEAR**2 - R**2))
        th = np.random.rand(n)*2*np.pi
        return np.column_stack([rr*np.cos(th), rr*np.sin(th)])

    def sample_points(n_pde, n_bc):
        n_near = int(NEAR_FRAC*n_pde); n_full = n_pde - n_near
        xy = sample_fluid(n_full)
        if n_near > 0:
            xy = np.vstack([xy, sample_near(n_near)])
        tt = np.random.rand(len(xy))*T
        pde = np.column_stack([xy, tt])
        ic = np.column_stack([IC_XY, IC_T]); ic_uv = IC_UV
        yb = Y0 + np.random.rand(n_bc)*(Y1 - Y0); tb = np.random.rand(n_bc)*T
        inlet = np.column_stack([np.full(n_bc, X0), yb, tb])
        xb = X0 + np.random.rand(n_bc)*(X1 - X0); tb2 = np.random.rand(n_bc)*T
        sgn = np.where(np.random.rand(n_bc) > 0.5, Y1, Y0)
        walls = np.column_stack([xb, sgn, tb2])
        th = np.random.rand(n_bc)*2*np.pi; tb3 = np.random.rand(n_bc)*T
        cyl = np.column_stack([R*np.cos(th), R*np.sin(th), tb3])
        return dict(pde=pde, ic=ic, ic_uv=ic_uv, inlet=inlet, walls=walls, cyl=cyl)

    # --------------------------------- network (32x3 streamfunction) + theta=log nu
    class FourierFeatures(nn.Module):
        """Random Fourier feature encoding, sin/cos(2*pi*B*x) with fixed B.

        Used by the bundle-C architecture ablation (referee R3.4). B is drawn once
        and registered as a buffer, so it is part of the seeded initialization and
        is not re-drawn between steps.
        """
        def __init__(self, in_dim, n_features, scale):
            super().__init__()
            self.register_buffer("B", torch.randn(in_dim, n_features) * scale)
        def forward(self, x):
            proj = 2.0 * np.pi * (x @ self.B)
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        @property
        def out_dim(self):
            return 2 * self.B.shape[1]

    class Net(nn.Module):
        def __init__(self, h=HIDDEN, L=LAYERS, nu0=NU0):
            super().__init__()
            enc = getattr(cfg, "pinn_encoding", "none").lower()
            if enc == "fourier":
                self.encode = FourierFeatures(3, cfg.fourier_features, cfg.fourier_scale)
                in_dim = self.encode.out_dim
            elif enc == "none":
                self.encode = None
                in_dim = 3
            else:
                raise ValueError(f"unknown pinn_encoding {enc!r}")
            layers = [nn.Linear(in_dim, h), nn.Tanh()]
            for _ in range(L-1):
                layers += [nn.Linear(h, h), nn.Tanh()]
            layers += [nn.Linear(h, 2)]
            self.net = nn.Sequential(*layers)
            self.theta = nn.Parameter(torch.tensor(float(np.log(nu0))))   # nu=exp(theta)>0
        def nu(self):
            return torch.exp(self.theta)
        def raw(self, x, y, t):
            xs = (x - X0)/(X1 - X0)*2 - 1; ys = y/Y1; ts = t/T*2 - 1
            inp = torch.stack([xs, ys, ts], dim=-1)
            if self.encode is not None:
                inp = self.encode(inp)
            out = self.net(inp)
            return out[..., 0], out[..., 1]

    def velocity(model, x, y, t):
        psi, p = model.raw(x, y, t)
        u = grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
        return u, v, p

    def tens(a, rg=False): return torch.tensor(a, device=DEV, requires_grad=rg)

    W_IC, W_BC, W_PDE, W_DATA = cfg.w_ic, cfg.w_bc, cfg.w_pde, cfg.w_data

    def loss_terms(model, pts, anchor=None, pts_w=None, return_tensors=False):
        """Composite loss.

        ``return_tensors`` additionally yields the individual differentiable terms,
        which gradient-norm balancing needs (it differentiates each one separately).
        ``anchor`` supplies the pseudo-time relaxation: the momentum residuals are
        augmented with ``pts_w * (u - u_prev)`` against a frozen earlier snapshot of
        the model, which damps the stiff early transient. Only the residual is
        anchored -- the data term keeps pulling on the real observations.
        """
        nu = model.nu()
        x = tens(pts['pde'][:, 0], True); y = tens(pts['pde'][:, 1], True); t = tens(pts['pde'][:, 2], True)
        u, v, p = velocity(model, x, y, t); o = torch.ones_like(u)
        ux = grad(u, x, o, create_graph=True)[0]; uy = grad(u, y, o, create_graph=True)[0]
        ut = grad(u, t, o, create_graph=True)[0]
        vx = grad(v, x, o, create_graph=True)[0]; vy = grad(v, y, o, create_graph=True)[0]
        vt = grad(v, t, o, create_graph=True)[0]
        uxx = grad(ux, x, o, create_graph=True)[0]; uyy = grad(uy, y, o, create_graph=True)[0]
        vxx = grad(vx, x, o, create_graph=True)[0]; vyy = grad(vy, y, o, create_graph=True)[0]
        px = grad(p, x, o, create_graph=True)[0]; py = grad(p, y, o, create_graph=True)[0]
        ru = ut + u*ux + v*uy + px - nu*(uxx + uyy)
        rv = vt + u*vx + v*vy + py - nu*(vxx + vyy)
        if anchor is not None and pts_w is not None and max(pts_w) > 0.0:
            # u_prev = d(psi_prev)/dy still needs autograd through the *inputs*, so
            # this cannot run under no_grad. The snapshot's parameters have
            # requires_grad=False and the result is detached, so no gradient reaches
            # the anchor -- it acts as a constant target.
            u_prev, v_prev, _ = velocity(anchor, x, y, t)
            ru = ru + pts_w[0] * (u - u_prev.detach())
            rv = rv + pts_w[1] * (v - v_prev.detach())
        l_pde = torch.mean(ru**2 + rv**2)
        xi = tens(pts['ic'][:, 0], True); yi = tens(pts['ic'][:, 1], True); ti = tens(pts['ic'][:, 2], True)
        ui, vi, _ = velocity(model, xi, yi, ti); tgt = tens(pts['ic_uv'])
        l_ic = torch.mean((ui - tgt[:, 0])**2 + (vi - tgt[:, 1])**2)
        def vel(arr):
            xb = tens(arr[:, 0], True); yb = tens(arr[:, 1], True); tb = tens(arr[:, 2], True)
            return velocity(model, xb, yb, tb)
        uI, vI, _ = vel(pts['inlet']); l_in = torch.mean((uI - 1.0)**2 + vI**2)
        uC, vC, _ = vel(pts['cyl']);   l_cy = torch.mean(uC**2 + vC**2)
        _, vW, _  = vel(pts['walls']); l_wl = torch.mean(vW**2)
        l_bc = l_in + l_cy + l_wl
        xd = tens(OBS_X, True); yd = tens(OBS_Y, True); td = tens(OBS_TT, True)
        ud, vd, _ = velocity(model, xd, yd, td)
        l_data = torch.mean((ud - tens(OBS_U))**2 + (vd - tens(OBS_V))**2)
        total = W_PDE*l_pde + W_IC*l_ic + W_BC*l_bc + W_DATA*l_data
        comp = dict(pde=l_pde.item(), ic=l_ic.item(), bc=l_bc.item(),
                    data=l_data.item(), nu=nu.item())
        if return_tensors:
            return (total, comp, dict(pde=l_pde, ic=l_ic, bc=l_bc, data=l_data),
                    dict(res=(ru, rv), sol=(u, v)))
        return total, comp

    EVAL_XY = sample_fluid(20000)
    EVAL_REF = eval_uv(rf_lin, rf_nea, EVAL_XY)
    def rel_l2_T(model):
        x = tens(EVAL_XY[:, 0], True); y = tens(EVAL_XY[:, 1], True); t = tens(np.full(len(EVAL_XY), T), True)
        u, v, _ = velocity(model, x, y, t)
        up = np.column_stack([u.detach().cpu().numpy(), v.detach().cpu().numpy()])
        return float(np.sqrt(np.sum((up-EVAL_REF)**2))/np.sqrt(np.sum(EVAL_REF**2)))

    class JaxpiNet(nn.Module):
        """Gated modified MLP + trainable theta = log nu, after jaxpi2.

        Same streamfunction-pressure output and the same nu parameterization as
        Net, so the two setups differ only in architecture and training, not in
        what is being solved for.
        """
        def __init__(self, h=HIDDEN, L=LAYERS, nu0=NU0):
            super().__init__()
            fe = {"embed_dim": h, "embed_scale": 1.0} if cfg.jaxpi_fourier else None
            if cfg.jaxpi_arch.lower() == "plain":
                # Vanilla arm: the paper's own tanh MLP, so the only difference
                # from the published run is the optimizer.
                self.net = PlainMlp(3, 2, hidden_dim=h, num_layers=L,
                                    activation="tanh", fourier_emb=fe)
            else:
                self.net = ModifiedMlp(3, 2, hidden_dim=h, num_layers=L,
                                       activation=cfg.jaxpi_activation, fourier_emb=fe)
            self.theta = nn.Parameter(torch.tensor(float(np.log(nu0))))
        def nu(self):
            return torch.exp(self.theta)
        def raw(self, x, y, t):
            xs = (x - X0)/(X1 - X0)*2 - 1; ys = y/Y1; ts = t/T*2 - 1
            out = self.net(torch.stack([xs, ys, ts], dim=-1))
            return out[..., 0], out[..., 1]

    if cfg.pinn_setup.lower() in ("jaxpi", "modern", "vanilla_adam"):
        return _train_jaxpi(cfg, JaxpiNet, loss_terms, rel_l2_T, sample_points,
                            NU_TRUE, NU0, DEV)

    # ----------------------------------------- cold start: Adam -> SSBroyden2
    model = Net(HIDDEN, LAYERS, nu0=NU0).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print("cold start: nu0=%.4f (theta0=%.4f), %d params" % (NU0, model.theta.item(), nparam), flush=True)

    N_PDE, N_BC = cfg.n_pde, cfg.n_bc
    hist = {'it': [], 'loss': [], 'pde': [], 'ic': [], 'bc': [], 'data': [],
            'err': [], 'nu': [], 'phase': []}

    ADAM_BLOCKS, ADAM_PER = cfg.adam_blocks, cfg.adam_per
    opt = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    t_start = time.time(); t0 = time.time(); git = 0
    for b in range(ADAM_BLOCKS):
        pts = sample_points(N_PDE, N_BC)
        for _ in range(ADAM_PER):
            opt.zero_grad()
            loss, comp = loss_terms(model, pts); loss.backward(); opt.step()
            git += 1
            if git % cfg.print_every == 0:
                err = rel_l2_T(model)
                hist['it'].append(git); hist['loss'].append(loss.item())
                hist['pde'].append(comp['pde']); hist['ic'].append(comp['ic'])
                hist['bc'].append(comp['bc']); hist['data'].append(comp['data'])
                hist['err'].append(err); hist['nu'].append(comp['nu']); hist['phase'].append('adam')
        print("  adam blk %d: loss=%.3e pde=%.2e data=%.2e nu=%.6f relnu=%.3e rel_L2(T)=%.3e"
              % (b, loss.item(), comp['pde'], comp['data'], comp['nu'],
                 abs(comp['nu']-NU_TRUE)/NU_TRUE, hist['err'][-1]), flush=True)
    print("Adam done %.1fs nu=%.6f" % (time.time()-t0, model.nu().item()), flush=True)

    def get_params():
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
    def set_params(flat):
        i = 0
        for p in model.parameters():
            n = p.numel(); p.data = tens(flat[i:i+n].reshape(p.shape)); i += n

    N_RESTARTS, MAXITER = cfg.n_restarts_inv, cfg.maxiter
    # Optional plateau stop across restarts. The published configuration runs a
    # fixed 10 restarts and its loss is still descending when it stops, so it is a
    # fixed-budget result rather than a converged one. Setting
    # inv_restart_plateau > 0 keeps restarting until the loss stops improving,
    # which is what the "paper_converged" arm of bundle C uses.
    plateau_n = getattr(cfg, "inv_restart_plateau", 0)
    plateau_rtol = getattr(cfg, "inv_restart_rtol", 1e-6)
    best_loss, best_r, restart_stop = float("inf"), 0, "maxrestarts"
    for r in range(N_RESTARTS):
        pts = sample_points(N_PDE, N_BC)
        cur = {}
        def objective(flat):
            set_params(flat)
            for p in model.parameters():
                if p.grad is not None: p.grad = None
            loss, comp = loss_terms(model, pts); loss.backward()
            g = np.concatenate([(p.grad if p.grad is not None else torch.zeros_like(p))
                                .detach().cpu().numpy().ravel() for p in model.parameters()])
            cur['loss'] = loss.item(); cur['comp'] = comp
            return loss.item(), g
        def callback(xk):
            nonlocal git
            git += 1
            if git % cfg.print_every == 0:
                set_params(xk)
                L, comp = loss_terms(model, pts); L = L.item(); err = rel_l2_T(model)
                hist['it'].append(git); hist['loss'].append(L); hist['pde'].append(comp['pde'])
                hist['ic'].append(comp['ic']); hist['bc'].append(comp['bc'])
                hist['data'].append(comp['data']); hist['err'].append(err)
                hist['nu'].append(comp['nu']); hist['phase'].append('ssb')
                print("  it %5d (r%d): loss=%.3e pde=%.2e data=%.2e nu=%.6f relnu=%.3e rel_L2(T)=%.3e"
                      % (git, r, L, comp['pde'], comp['data'], comp['nu'],
                         abs(comp['nu']-NU_TRUE)/NU_TRUE, err), flush=True)
        res = minimize(objective, get_params(), method='BFGS', jac=True, callback=callback,
                       options={'maxiter': MAXITER, 'gtol': cfg.gtol_inv, 'disp': False,
                                'method_bfgs': 'SSBroyden2'})
        set_params(res.x)
        nu_r = model.nu().item()
        print(" -- restart %d done: nit=%d loss=%.3e nu=%.6f relnu=%.3e rel_L2(T)=%.3e"
              % (r, res.nit, cur.get('loss', np.nan), nu_r, abs(nu_r-NU_TRUE)/NU_TRUE,
                 rel_l2_T(model)), flush=True)
        if plateau_n:
            lr_now = float(cur.get('loss', np.nan))
            if np.isfinite(lr_now) and lr_now < best_loss * (1.0 - plateau_rtol):
                best_loss, best_r = lr_now, r
            elif r - best_r >= plateau_n:
                restart_stop = "plateau"
                print(" -- restart plateau: no improvement in %d restarts, stopping"
                      % plateau_n, flush=True)
                break
    print("SSBroyden done %.1fs (stop=%s)" % (time.time()-t0, restart_stop), flush=True)
    runtime = time.time() - t_start

    nu_final = model.nu().item()
    print("\n=== INVERSE RESULT (%dx%d, %d probes, nu0=%.4f) ===" % (HIDDEN, LAYERS, NP, NU0), flush=True)
    print("nu_true = %.8f  Re_true = %.4f" % (NU_TRUE, 1.0/NU_TRUE), flush=True)
    print("nu_rec  = %.8f  Re_rec  = %.4f" % (nu_final, 1.0/nu_final), flush=True)
    print("rel nu error = %.4e" % (abs(nu_final-NU_TRUE)/NU_TRUE), flush=True)

    # ---------------------------------------------------------- training figure
    os.makedirs(cfg.fig_dir, exist_ok=True)
    it = np.array(hist['it']); nu_h = np.array(hist['nu'])
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].semilogy(it, hist['loss'], '-', label='total loss')
    ax[0].semilogy(it, hist['pde'], '--', alpha=0.7, label='PDE residual')
    ax[0].semilogy(it, hist['data'], ':', alpha=0.9, label='probe-data misfit')
    ax[0].set_xlabel('iteration'); ax[0].set_ylabel('loss'); ax[0].legend()
    ax[0].set_title('Inverse PINN training loss'); ax[0].grid(alpha=0.3, which='both')
    ax[1].plot(it, nu_h, '-'); ax[1].axhline(NU_TRUE, color='r', ls='--', lw=1, label='nu_true')
    ax[1].axhline(NU0, color='gray', ls=':', lw=1, label='nu0=%.3f' % NU0)
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('nu'); ax[1].legend()
    ax[1].set_title('Viscosity convergence'); ax[1].grid(alpha=0.3)
    ax[2].semilogy(it, np.abs(nu_h-NU_TRUE)/NU_TRUE, '-')
    ax[2].set_xlabel('iteration'); ax[2].set_ylabel('rel nu error')
    ax[2].set_title('Inverse convergence (runtime %ds)' % int(runtime)); ax[2].grid(alpha=0.3, which='both')
    na = hist['phase'].count('adam')
    if 0 < na < len(it):
        for a in ax: a.axvline(it[na-1], color='k', lw=0.8, alpha=0.4, ls=':')
    fig.tight_layout(); fig.savefig(os.path.join(cfg.fig_dir, "pinn_inv_train.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------- save
    os.makedirs(cfg.hist_dir, exist_ok=True)
    np.savez(cfg.pinn_inv_npz, T=T, t0a=t0a, nu_true=NU_TRUE, nu_rec=nu_final, nu0=NU0,
             it=it, loss=np.array(hist['loss']), pde=np.array(hist['pde']),
             ic=np.array(hist['ic']), bc=np.array(hist['bc']), data=np.array(hist['data']),
             err=np.array(hist['err']), nu=nu_h, phase=np.array(hist['phase']),
             probe_xy=probe_xy, runtime_sec=runtime, cfg_snapshot=str(cfg.snapshot()))
    torch.save(model.state_dict(), cfg.pinn_inv_model)
    print("saved %s + %s (%.1fs)" % (cfg.pinn_inv_npz, cfg.pinn_inv_model, runtime), flush=True)

    return dict(model=model, nu_rec=nu_final, nu_true=NU_TRUE,
                rel_err=abs(nu_final-NU_TRUE)/NU_TRUE, runtime_sec=runtime,
                hist=hist, T=T, t0a=t0a, probe_xy=probe_xy,
                restart_stop=restart_stop, n_restarts_run=r + 1)


# ============================================================================
# jaxpi2-style training loop (bundle C, second arm)
# ============================================================================

def _train_jaxpi(cfg, NetCls, loss_terms, rel_l2_T, sample_points,
                 NU_TRUE, NU0, DEV):
    """SOAP + gradient-norm balancing + pseudo-time relaxation.

    Reimplements the recipe from github.com/sifanexisted/jaxpi2 in PyTorch. The
    reference is jax/flax and targets forward problems; the one adaptation for our
    inverse setting is that only the PDE residual is anchored in pseudo-time, so
    the sparse probe observations always pull on the true solution.

    Sized to the paper rather than to the reference example -- see cylinder_config.
    """
    import os
    import time

    import matplotlib.pyplot as plt

    from common.gradnorm import GradNormWeighter
    from common.pseudotime import PseudoTimeStepper
    from common.soap import SOAP

    model = NetCls().to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print("setup=%s arch=%s opt=%s gradnorm=%s pseudotime=%s | %d params, nu0=%.4f"
          % (cfg.pinn_setup, cfg.jaxpi_arch, cfg.jaxpi_optimizer,
             cfg.jaxpi_gradnorm, cfg.jaxpi_pseudotime, nparam, NU0), flush=True)

    if cfg.jaxpi_optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.jaxpi_lr)
    else:
        opt = SOAP(model.parameters(), lr=cfg.jaxpi_lr,
                   precondition_frequency=5,      # cyl_pinn_pt.py default
                   precondition_1d=True, weight_decay=0.0)
    # Linear warmup then exponential decay, as in the reference config.
    def lr_at(step):
        if step < cfg.jaxpi_warmup_steps:
            return (step + 1) / max(1, cfg.jaxpi_warmup_steps)
        k = (step - cfg.jaxpi_warmup_steps) / cfg.jaxpi_decay_steps
        return cfg.jaxpi_decay_rate ** k
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    keys = ("pde", "ic", "bc", "data")
    weighter = (GradNormWeighter(keys, model.parameters(),
                                 update_every=cfg.jaxpi_gradnorm_every,
                                 momentum=cfg.jaxpi_gradnorm_momentum)
                if cfg.jaxpi_gradnorm else None)
    if weighter is not None:
        # Start from the paper's weights so step 0 is the same problem.
        weighter.weights.update(dict(pde=cfg.w_pde, ic=cfg.w_ic,
                                     bc=cfg.w_bc, data=cfg.w_data))

    stepper = (PseudoTimeStepper(n_components=2,
                                 update_every=cfg.jaxpi_pts_every,
                                 momentum=cfg.jaxpi_pts_momentum)
               if cfg.jaxpi_pseudotime else None)
    if stepper is not None:
        stepper.refresh(model)

    hist = {'it': [], 'loss': [], 'pde': [], 'ic': [], 'bc': [], 'data': [],
            'err': [], 'nu': [], 'phase': [], 'w': [], 'theta': []}
    pts = sample_points(cfg.n_pde, cfg.n_bc)
    t_start = time.time()

    for step in range(cfg.jaxpi_steps):
        if step > 0 and step % cfg.jaxpi_resample_every == 0:
            pts = sample_points(cfg.n_pde, cfg.n_bc)

        theta = stepper.theta if (stepper is not None and stepper.active) else None
        prev = stepper.prev if stepper is not None else None

        opt.zero_grad(set_to_none=True)
        _, comp, terms, raw = loss_terms(model, pts, anchor=prev, pts_w=theta,
                                         return_tensors=True)

        if weighter is not None and weighter.should_update(step):
            weighter.update(terms)
        total = (weighter.weighted_total(terms) if weighter is not None
                 else cfg.w_pde*terms['pde'] + cfg.w_ic*terms['ic']
                      + cfg.w_bc*terms['bc'] + cfg.w_data*terms['data'])
        total.backward()
        opt.step()
        sched.step()

        if stepper is not None:
            # theta = ||dR|| / ||du|| between the current and previous iterate,
            # recomputed on a cadence because it costs two extra residual passes.
            if stepper.should_update(step):
                with torch.enable_grad():
                    _, _, _, raw_now = loss_terms(model, pts, return_tensors=True)
                    _, _, _, raw_prev = loss_terms(stepper.prev, pts, return_tensors=True)
                res_now = [r.detach() for r in raw_now['res']]
                res_prev = [r.detach() for r in raw_prev['res']]
                sol_now = [u.detach() for u in raw_now['sol']]
                sol_prev = [u.detach() for u in raw_prev['sol']]
                losses = [float((r**2).mean()) for r in res_now]
                stepper.update_theta(res_now, res_prev, sol_now, sol_prev, losses)
            # The previous iterate is refreshed EVERY step: that is what makes the
            # relaxation a pseudo-time derivative rather than a stale anchor.
            stepper.refresh(model)

        if (step + 1) % cfg.print_every == 0 or step == 0:
            err = rel_l2_T(model)
            hist['it'].append(step + 1); hist['loss'].append(float(total.detach()))
            for k in ('pde', 'ic', 'bc', 'data'):
                hist[k].append(comp[k])
            hist['err'].append(err); hist['nu'].append(comp['nu'])
            hist['phase'].append('jaxpi')
            hist['w'].append(dict(weighter.weights) if weighter else {})
            hist['theta'].append(list(stepper.theta) if stepper else [])
            if (step + 1) % (20 * cfg.print_every) == 0 or step == 0:
                wtxt = (" w=" + ",".join(f"{k}:{v:.2e}" for k, v in weighter.weights.items())
                        ) if weighter else ""
                ttxt = (" th=" + ",".join(f"{t:.2e}" for t in stepper.theta)) if stepper else ""
                print("  it %6d: loss=%.3e pde=%.2e data=%.2e nu=%.6f relnu=%.3e "
                      "rel_L2(T)=%.3e lr=%.2e%s%s"
                      % (step + 1, hist['loss'][-1], comp['pde'], comp['data'],
                         comp['nu'], abs(comp['nu']-NU_TRUE)/NU_TRUE, err,
                         opt.param_groups[0]['lr'], ttxt, wtxt), flush=True)

    runtime = time.time() - t_start
    nu_final = float(model.nu().item())
    print("jaxpi done %.1fs nu=%.8f rel=%.3e"
          % (runtime, nu_final, abs(nu_final-NU_TRUE)/NU_TRUE), flush=True)

    os.makedirs(cfg.fig_dir, exist_ok=True)
    os.makedirs(cfg.hist_dir, exist_ok=True)
    it = np.array(hist['it'])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].semilogy(it, hist['loss'], label='total')
    ax[0].semilogy(it, hist['pde'], '--', alpha=0.7, label='PDE residual')
    ax[0].semilogy(it, hist['data'], ':', alpha=0.9, label='probe-data misfit')
    ax[0].set_xlabel('iteration'); ax[0].set_ylabel('loss'); ax[0].legend()
    ax[0].set_title('jaxpi setup: training loss'); ax[0].grid(alpha=0.3, which='both')
    ax[1].plot(it, hist['nu']); ax[1].axhline(NU_TRUE, color='r', ls='--', lw=1)
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('nu'); ax[1].grid(alpha=0.3)
    ax[1].set_title('viscosity convergence')
    ax[2].semilogy(it, np.abs(np.array(hist['nu'])-NU_TRUE)/NU_TRUE)
    ax[2].set_xlabel('iteration'); ax[2].set_ylabel('rel nu error')
    ax[2].set_title('runtime %ds' % int(runtime)); ax[2].grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.fig_dir, "pinn_inv_train_jaxpi.png"), dpi=120)
    plt.close(fig)

    np.savez(cfg.pinn_inv_npz, nu_true=NU_TRUE, nu_rec=nu_final, nu0=NU0,
             it=it, loss=np.array(hist['loss']), pde=np.array(hist['pde']),
             ic=np.array(hist['ic']), bc=np.array(hist['bc']),
             data=np.array(hist['data']), err=np.array(hist['err']),
             nu=np.array(hist['nu']), runtime_sec=runtime,
             setup='jaxpi', cfg_snapshot=str(cfg.snapshot()))
    torch.save(model.state_dict(), cfg.pinn_inv_model)

    return dict(model=model, nu_rec=nu_final, nu_true=NU_TRUE,
                rel_err=abs(nu_final-NU_TRUE)/NU_TRUE, runtime_sec=runtime,
                hist=hist, T=None, t0a=None, probe_xy=None,
                gradnorm=(weighter.state() if weighter else None),
                pseudotime=(stepper.state() if stepper else None))
