"""Forward PINN for the cylinder flow (space-time MLP -> psi, p; hard div-free).

Canonical successor of cylinder_pinn_fwd4.py, refactored into a single
train(cfg) entry point that reads every control flag from a CylinderRunConfig
(net 32x3, NO near-cylinder oversampling, Re from cfg). It learns the forward
field over one shedding period from the FEM IC (saturated.npz), with Adam warmup
-> SSBroyden2 BFGS, monitoring loss components and rel-L2 vs FEM every cfg.print_every
iters. Saves cfg.pinn_fwd_npz + cfg.pinn_fwd_model + a training-curve figure.
Run in env opt_2nd (GPU). See methodology_Cylinder.md.
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

torch.set_default_dtype(torch.float64)


def train(cfg, profile_only=False):
    DEV = cfg.device
    torch.manual_seed(cfg.seed_fwd); np.random.seed(cfg.seed_fwd)
    print("device =", DEV, flush=True)

    X0, X1, Y0, Y1 = cfg.x0, cfg.x1, cfg.y0, cfg.y1
    R, NU = cfg.R, cfg.nu_true
    MARGIN, R_NEAR, NEAR_FRAC = cfg.margin, cfg.r_near, cfg.near_frac
    HIDDEN, LAYERS = cfg.hidden, cfg.layers
    T = cfg.T_obs

    # ------------------------------------------------------------------ IC/ref
    cyl_cfg = cfg.solver_cfg("prod")
    mesh, _, _ = create_solver(cyl_cfg)
    nps = mesh.n_p2; XY = mesh.coords_p2
    with np.load(cfg.saturated_path, mmap_mode='r') as z:
        times = np.array(z['times']); snaps = z['snaps']
        t0a = float(z['t0a'])
        snap0 = np.array(z['u0'])                                # exact IC
        i1 = int(np.argmin(np.abs(times - (t0a + T))))
        snap1 = np.array(snaps[i1])                              # ref at t=T
    print("IC t=%.3f ref t=%.3f T=%.4f (n_p2=%d)" % (t0a, times[i1], T, nps), flush=True)

    def interp_pair(snap):
        uv = np.column_stack([snap[:nps], snap[nps:2*nps]])
        return LinearNDInterpolator(XY, uv), NearestNDInterpolator(XY, uv)
    rf_lin, rf_nea = interp_pair(snap1)

    def eval_uv(lin, nea, pts):
        v = lin(pts); bad = np.isnan(v[:, 0])
        if bad.any(): v[bad] = nea(pts[bad])
        return v

    IC_XY = XY.copy()
    IC_UV = np.column_stack([snap0[:nps], snap0[nps:2*nps]])
    IC_T = np.zeros(len(IC_XY))
    print("IC = all %d mesh nodes (exact)" % len(IC_XY), flush=True)

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

    # ------------------------------------------------------------- network/loss
    class Net(nn.Module):
        def __init__(self, h=HIDDEN, L=LAYERS):
            super().__init__()
            layers = [nn.Linear(3, h), nn.Tanh()]
            for _ in range(L-1):
                layers += [nn.Linear(h, h), nn.Tanh()]
            layers += [nn.Linear(h, 2)]
            self.net = nn.Sequential(*layers)
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
        def raw(self, x, y, t):
            xs = (x - X0)/(X1 - X0)*2 - 1; ys = y/Y1; ts = t/T*2 - 1
            out = self.net(torch.stack([xs, ys, ts], dim=-1))
            return out[..., 0], out[..., 1]

    def velocity(model, x, y, t):
        psi, p = model.raw(x, y, t)
        u = grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        v = -grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
        return u, v, p

    def tens(a, rg=False): return torch.tensor(a, device=DEV, requires_grad=rg)

    def sync_device():
        if DEV == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    W_IC, W_BC, W_PDE = cfg.w_ic, cfg.w_bc, cfg.w_pde

    def loss_terms(model, pts, profile=False):
        t_prev = None
        timing = None
        if profile:
            timing = {}
            sync_device()
            t_prev = time.perf_counter()

        x = tens(pts['pde'][:, 0], True); y = tens(pts['pde'][:, 1], True); t = tens(pts['pde'][:, 2], True)
        u, v, p = velocity(model, x, y, t); o = torch.ones_like(u)
        ux = grad(u, x, o, create_graph=True)[0]; uy = grad(u, y, o, create_graph=True)[0]
        ut = grad(u, t, o, create_graph=True)[0]
        vx = grad(v, x, o, create_graph=True)[0]; vy = grad(v, y, o, create_graph=True)[0]
        vt = grad(v, t, o, create_graph=True)[0]
        uxx = grad(ux, x, o, create_graph=True)[0]; uyy = grad(uy, y, o, create_graph=True)[0]
        vxx = grad(vx, x, o, create_graph=True)[0]; vyy = grad(vy, y, o, create_graph=True)[0]
        px = grad(p, x, o, create_graph=True)[0]; py = grad(p, y, o, create_graph=True)[0]
        ru = ut + u*ux + v*uy + px - NU*(uxx + uyy)
        rv = vt + u*vx + v*vy + py - NU*(vxx + vyy)
        l_pde = torch.mean(ru**2 + rv**2)
        if profile:
            sync_device()
            t_now = time.perf_counter()
            timing['pde_ms'] = 1e3*(t_now - t_prev)
            t_prev = t_now

        xi = tens(pts['ic'][:, 0], True); yi = tens(pts['ic'][:, 1], True); ti = tens(pts['ic'][:, 2], True)
        ui, vi, _ = velocity(model, xi, yi, ti); tgt = tens(pts['ic_uv'])
        l_ic = torch.mean((ui - tgt[:, 0])**2 + (vi - tgt[:, 1])**2)
        if profile:
            sync_device()
            t_now = time.perf_counter()
            timing['ic_ms'] = 1e3*(t_now - t_prev)
            t_prev = t_now

        def vel(arr):
            xb = tens(arr[:, 0], True); yb = tens(arr[:, 1], True); tb = tens(arr[:, 2], True)
            return velocity(model, xb, yb, tb)

        uI, vI, _ = vel(pts['inlet']); l_in = torch.mean((uI - 1.0)**2 + vI**2)
        if profile:
            sync_device()
            t_now = time.perf_counter()
            timing['inlet_ms'] = 1e3*(t_now - t_prev)
            t_prev = t_now

        uC, vC, _ = vel(pts['cyl']);   l_cy = torch.mean(uC**2 + vC**2)
        if profile:
            sync_device()
            t_now = time.perf_counter()
            timing['cyl_ms'] = 1e3*(t_now - t_prev)
            t_prev = t_now

        _, vW, _  = vel(pts['walls']); l_wl = torch.mean(vW**2)
        if profile:
            sync_device()
            t_now = time.perf_counter()
            timing['walls_ms'] = 1e3*(t_now - t_prev)
            t_prev = t_now

        l_bc = l_in + l_cy + l_wl
        total = W_PDE*l_pde + W_IC*l_ic + W_BC*l_bc
        comp = dict(
            pde=l_pde.item(), ic=l_ic.item(), bc=l_bc.item(),
            inlet=l_in.item(), cyl=l_cy.item(), walls=l_wl.item(),
            n_pde=len(pts['pde']), n_ic=len(pts['ic']),
            n_inlet=len(pts['inlet']), n_cyl=len(pts['cyl']), n_walls=len(pts['walls'])
        )
        if profile:
            timing['bc_ms'] = timing['inlet_ms'] + timing['cyl_ms'] + timing['walls_ms']
            timing['loss_ms'] = timing['pde_ms'] + timing['ic_ms'] + timing['bc_ms']
            comp.update(timing)
        return total, comp

    EVAL_XY = sample_fluid(20000)
    EVAL_REF = eval_uv(rf_lin, rf_nea, EVAL_XY)
    def rel_l2_T(model):
        x = tens(EVAL_XY[:, 0], True); y = tens(EVAL_XY[:, 1], True); t = tens(np.full(len(EVAL_XY), T), True)
        u, v, _ = velocity(model, x, y, t)
        up = np.column_stack([u.detach().cpu().numpy(), v.detach().cpu().numpy()])
        return float(np.sqrt(np.sum((up-EVAL_REF)**2))/np.sqrt(np.sum(EVAL_REF**2)))

    # ------------------------------------------------- training: Adam -> SSBroyden2
    model = Net().to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print("fresh net %dx%d, params=%d" % (HIDDEN, LAYERS, nparam), flush=True)

    hist = {
        'phase': [], 'loss': [], 'pde': [], 'ic': [], 'bc': [], 'err': [],
        'pde_ms': [], 'ic_ms': [], 'bc_ms': [], 'inlet_ms': [], 'cyl_ms': [], 'walls_ms': [], 'loss_ms': []
    }
    timing_log_once = {'done': False}

    def maybe_print_timing(comp, indent="             "):
        if timing_log_once['done']:
            return
        print("%sms: loss=%.1f pde=%.1f ic=%.1f bc=%.1f [inlet=%.1f cyl=%.1f walls=%.1f]"
              % (indent, comp['loss_ms'], comp['pde_ms'], comp['ic_ms'], comp['bc_ms'],
                 comp['inlet_ms'], comp['cyl_ms'], comp['walls_ms']), flush=True)
        timing_log_once['done'] = True

    def record(phase, loss, comp, err):
        hist['phase'].append(phase); hist['loss'].append(loss)
        hist['pde'].append(comp['pde']); hist['ic'].append(comp['ic'])
        hist['bc'].append(comp['bc']); hist['err'].append(err)
        hist['pde_ms'].append(comp.get('pde_ms', np.nan))
        hist['ic_ms'].append(comp.get('ic_ms', np.nan))
        hist['bc_ms'].append(comp.get('bc_ms', np.nan))
        hist['inlet_ms'].append(comp.get('inlet_ms', np.nan))
        hist['cyl_ms'].append(comp.get('cyl_ms', np.nan))
        hist['walls_ms'].append(comp.get('walls_ms', np.nan))
        hist['loss_ms'].append(comp.get('loss_ms', np.nan))

    N_PDE, N_BC = cfg.n_pde, cfg.n_bc
    ADAM_EPOCHS, ADAM_REC = cfg.adam_epochs, cfg.adam_rec
    print("loss points: PDE=%d  IC=%d  inlet=%d  cyl=%d  walls=%d"
          % (N_PDE, len(IC_XY), N_BC, N_BC, N_BC), flush=True)
    if profile_only:
        pts = sample_points(N_PDE, N_BC)
        total, comp = loss_terms(model, pts, profile=True)
        total = total.item()
        print("profile only: loss=%.3e pde=%.2e ic=%.2e bc=%.2e"
              % (total, comp['pde'], comp['ic'], comp['bc']), flush=True)
        print("              ms: loss=%.1f pde=%.1f ic=%.1f bc=%.1f [inlet=%.1f cyl=%.1f walls=%.1f]"
              % (comp['loss_ms'], comp['pde_ms'], comp['ic_ms'], comp['bc_ms'],
                 comp['inlet_ms'], comp['cyl_ms'], comp['walls_ms']), flush=True)
        return dict(loss=total, components=comp, point_counts=dict(
            n_pde=N_PDE, n_ic=len(IC_XY), n_inlet=N_BC, n_cyl=N_BC, n_walls=N_BC
        ))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    t_start = time.time(); t0 = time.time()
    pts = sample_points(N_PDE, N_BC)
    for ep in range(ADAM_EPOCHS):
        if ep % 200 == 0:
            pts = sample_points(N_PDE, N_BC)
        opt.zero_grad()
        loss, comp = loss_terms(model, pts); loss.backward(); opt.step()
        if ep % ADAM_REC == 0 or ep == ADAM_EPOCHS-1:
            L_rec, comp_rec = loss_terms(model, pts, profile=True); L_rec = L_rec.item()
            err = rel_l2_T(model)
            record('adam', L_rec, comp_rec, err)
            if ep % 500 == 0 or ep == ADAM_EPOCHS-1:
                print("  adam %5d  loss=%.3e pde=%.2e ic=%.2e bc=%.2e  rel_L2(T)=%.4e"
                      % (ep, L_rec, comp_rec['pde'], comp_rec['ic'], comp_rec['bc'], err), flush=True)
                maybe_print_timing(comp_rec)
    print("Adam done %.1fs  rel_L2(T)=%.4e" % (time.time()-t0, rel_l2_T(model)), flush=True)

    def get_params():
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
    def set_params(flat):
        i = 0
        for p in model.parameters():
            n = p.numel(); p.data = tens(flat[i:i+n].reshape(p.shape)); i += n

    N_RESTARTS, MAXITER = cfg.n_restarts_fwd, cfg.maxiter
    gstate = {'it': 0}
    t0 = time.time()
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
            cur['loss'] = loss.item()
            return loss.item(), g
        def callback(xk):
            gstate['it'] += 1
            if gstate['it'] % cfg.print_every == 0:
                set_params(xk)
                L, comp = loss_terms(model, pts, profile=True); L = L.item()
                err = rel_l2_T(model)
                record('bfgs', L, comp, err)
                print("  it %5d (r%d): loss=%.3e pde=%.2e ic=%.2e bc=%.2e  rel_L2(T)=%.4e"
                      % (gstate['it'], r, L, comp['pde'], comp['ic'], comp['bc'], err), flush=True)
                maybe_print_timing(comp, indent="                 ")
        res = minimize(objective, get_params(), method='BFGS', jac=True, callback=callback,
                       options={'maxiter': MAXITER, 'gtol': cfg.gtol_fwd, 'disp': False,
                                'method_bfgs': 'SSBroyden2'})
        set_params(res.x)
        print(" -- restart %2d done: nit=%d loss=%.3e rel_L2(T)=%.4e"
              % (r, res.nit, cur.get('loss', np.nan), rel_l2_T(model)), flush=True)
    print("SSBroyden done %.1fs, %d total iters" % (time.time()-t0, gstate['it']), flush=True)
    runtime = time.time() - t_start

    # ---------------------------------------------------------- training figure
    os.makedirs(cfg.fig_dir, exist_ok=True)
    phase = np.array(hist['phase']); k = np.arange(len(phase))
    nb = int(np.argmax(phase == 'bfgs')) if 'bfgs' in phase else len(phase)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].semilogy(k, hist['loss'], '-', label='total loss')
    ax[0].semilogy(k, hist['pde'], '--', alpha=0.7, label='PDE residual')
    ax[0].semilogy(k, hist['ic'], ':', alpha=0.7, label='IC')
    ax[0].semilogy(k, hist['bc'], '-.', alpha=0.7, label='BC')
    ax[0].axvline(nb, color='r', lw=1, ls='--', alpha=0.6, label='Adam->SSBroyden')
    ax[0].set_xlabel('record step'); ax[0].set_ylabel('loss'); ax[0].legend()
    ax[0].set_title('Forward PINN training loss (%dx%d)' % (HIDDEN, LAYERS))
    ax[0].grid(alpha=0.3, which='both')
    ax[1].semilogy(k, hist['err'], 'o-', ms=3)
    ax[1].axvline(nb, color='r', lw=1, ls='--', alpha=0.6)
    ax[1].set_xlabel('record step'); ax[1].set_ylabel('rel L2 vs FEM at t=T')
    ax[1].set_title('Forward error during training (runtime %ds)' % int(runtime))
    ax[1].grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(os.path.join(cfg.fig_dir, "pinn_fwd_train.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------- error across the shedding period
    print("\n=== forward PINN vs FEM (rel L2 over fluid) across the period ===", flush=True)
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]; errs = []
    for fr in fracs:
        tt = fr*T; idx = int(np.argmin(np.abs(times - (t0a + tt))))
        snap = np.array(snaps[idx])
        li = LinearNDInterpolator(XY, np.column_stack([snap[:nps], snap[nps:2*nps]]))
        ne = NearestNDInterpolator(XY, np.column_stack([snap[:nps], snap[nps:2*nps]]))
        ref = eval_uv(li, ne, EVAL_XY)
        x = tens(EVAL_XY[:, 0], True); y = tens(EVAL_XY[:, 1], True); t = tens(np.full(len(EVAL_XY), tt), True)
        u, v, _ = velocity(model, x, y, t)
        up = np.column_stack([u.detach().cpu().numpy(), v.detach().cpu().numpy()])
        e = float(np.sqrt(np.sum((up-ref)**2))/np.sqrt(np.sum(ref**2))); errs.append(e)
        print("  t=%.3f (%.0f%%):  rel L2 = %.4e" % (tt, fr*100, e), flush=True)

    # ------------------------------------------------------------------- save
    os.makedirs(cfg.hist_dir, exist_ok=True)
    np.savez(cfg.pinn_fwd_npz, T=T, t0a=t0a, errs=np.array(errs), fracs=np.array(fracs),
             phase=phase, loss=np.array(hist['loss']), pde=np.array(hist['pde']),
             ic=np.array(hist['ic']), bc=np.array(hist['bc']), err=np.array(hist['err']),
             pde_ms=np.array(hist['pde_ms']), ic_ms=np.array(hist['ic_ms']),
             bc_ms=np.array(hist['bc_ms']), inlet_ms=np.array(hist['inlet_ms']),
             cyl_ms=np.array(hist['cyl_ms']), walls_ms=np.array(hist['walls_ms']),
             loss_ms=np.array(hist['loss_ms']),
             n_pde=N_PDE, n_ic=len(IC_XY), n_inlet=N_BC, n_cyl=N_BC, n_walls=N_BC,
             nparam=nparam, hidden=HIDDEN, layers=LAYERS, runtime_sec=runtime,
             cfg_snapshot=str(cfg.snapshot()))
    torch.save(model.state_dict(), cfg.pinn_fwd_model)
    print("\nsaved %s + %s. final rel_L2(t=T)=%.4e (%.1fs)"
          % (cfg.pinn_fwd_npz, cfg.pinn_fwd_model, errs[-1], runtime), flush=True)

    return dict(model=model, errs=np.array(errs), fracs=np.array(fracs),
                runtime_sec=runtime, hist=hist, T=T, t0a=t0a)
