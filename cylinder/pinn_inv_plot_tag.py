"""Plot the inverse-PINN result: (top) the viscosity-identification story
-- nu and rel-nu-error vs iteration with the Adam/SSBroyden phase split, plus
loss components; (bottom) the recovered field vs FEM at t=T with the wake
probes overlaid. Loads history/pinn_inv.npz + pinn_inv_model.pt -- no retraining.
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from cylinder_solver import CylinderConfig, create_solver

import sys
TAG = sys.argv[1] if len(sys.argv)>1 else ''
torch.set_default_dtype(torch.float64)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
X0, X1, Y0, Y1, R = -3.0, 10.0, -3.0, 3.0, 0.5

class Net(nn.Module):
    def __init__(self, h=32, L=3, T=6.0):
        super().__init__()
        self.T = T
        layers = [nn.Linear(3, h), nn.Tanh()]
        for _ in range(L-1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers += [nn.Linear(h, 2)]
        self.net = nn.Sequential(*layers)
        self.theta = nn.Parameter(torch.tensor(0.0))
    def raw(self, x, y, t):
        xs = (x - X0)/(X1 - X0)*2 - 1; ys = y/Y1; ts = t/self.T*2 - 1
        out = self.net(torch.stack([xs, ys, ts], dim=-1))
        return out[..., 0], out[..., 1]

def velocity(model, x, y, t):
    psi, p = model.raw(x, y, t)
    u =  grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
    v = -grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    return u, v

d = np.load('history/pinn_inv%s.npz' % TAG)
T, t0a = float(d['T']), float(d['t0a'])
NU_TRUE, NU_REC, NU0 = float(d['nu_true']), float(d['nu_rec']), float(d['nu0'])
it, nu_h, loss_h, pde_h, data_h = d['it'], d['nu'], d['loss'], d['pde'], d['data']
phase = d['phase']; na = int((phase == 'adam').sum())
model = Net(32, 3, T=T).to(DEV)
model.load_state_dict(torch.load('history/pinn_inv%s_model.pt' % TAG, map_location=DEV, weights_only=True))
model.eval()
print("loaded; nu_rec=%.6e theta=%.4f" % (np.exp(model.theta.item()), model.theta.item()), flush=True)

probe_xy = d['probe_xy']

# FEM ref at t=T
cfg = CylinderConfig(h_cyl=0.04, h_wake=0.08, h_far=0.5)
mesh, _, _ = create_solver(cfg)
nps = mesh.n_p2; XY = mesh.coords_p2
zt = np.load('history/validate.npz', mmap_mode='r'); times = np.array(zt['times'])
nx, ny = 320, 150
gx = np.linspace(X0, X1, nx); gy = np.linspace(Y0, Y1, ny)
GX, GY = np.meshgrid(gx, gy); pts = np.column_stack([GX.ravel(), GY.ravel()])
inside = (np.hypot(GX, GY) <= R + 1e-3)

def pinn_v(tt):
    x = torch.tensor(pts[:, 0], device=DEV, requires_grad=True)
    y = torch.tensor(pts[:, 1], device=DEV, requires_grad=True)
    t = torch.tensor(np.full(len(pts), tt), device=DEV, requires_grad=True)
    _, v = velocity(model, x, y, t)
    return v.detach().cpu().numpy().reshape(ny, nx)

def fem_v(tt):
    idx = int(np.argmin(np.abs(times - (t0a + tt))))
    snap = np.array(zt['snaps'][idx])
    uv = np.column_stack([snap[:nps], snap[nps:2*nps]])
    li = LinearNDInterpolator(XY, uv); ne = NearestNDInterpolator(XY, uv)
    w = li(pts); bad = np.isnan(w[:, 0])
    if bad.any(): w[bad] = ne(pts[bad])
    return w[:, 1].reshape(ny, nx)

fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

# --- top: the inversion story ---
axA = fig.add_subplot(gs[0, 0])
axA.plot(it, nu_h, '-', lw=1.5)
axA.axhline(NU_TRUE, color='r', ls='--', lw=1.2, label='nu_true=1/60')
axA.axhline(NU0, color='gray', ls=':', lw=1, label='nu0=1')
axA.axvline(it[na-1] if 0 < na < len(it) else it[0], color='k', lw=0.8, alpha=0.4, ls=':')
axA.set_yscale('log'); axA.set_xlabel('iteration'); axA.set_ylabel('nu (log)')
axA.set_title('Viscosity trajectory (Adam | SSBroyden)'); axA.legend(fontsize=8); axA.grid(alpha=0.3, which='both')

axB = fig.add_subplot(gs[0, 1])
axB.semilogy(it, np.abs(nu_h-NU_TRUE)/NU_TRUE, '-')
axB.axvline(it[na-1] if 0 < na < len(it) else it[0], color='k', lw=0.8, alpha=0.4, ls=':')
axB.set_xlabel('iteration'); axB.set_ylabel('rel nu error')
axB.set_title('Inverse convergence (final rel err %.1f%%)' % (100*abs(NU_REC-NU_TRUE)/NU_TRUE))
axB.grid(alpha=0.3, which='both')

axC = fig.add_subplot(gs[0, 2])
axC.semilogy(it, loss_h, '-', label='total')
axC.semilogy(it, pde_h, '--', alpha=0.7, label='PDE residual')
axC.semilogy(it, data_h, ':', alpha=0.9, label='probe-data misfit')
axC.axvline(it[na-1] if 0 < na < len(it) else it[0], color='k', lw=0.8, alpha=0.4, ls=':')
axC.set_xlabel('iteration'); axC.set_ylabel('loss'); axC.legend(fontsize=8)
axC.set_title('Loss components'); axC.grid(alpha=0.3, which='both')

# --- bottom: field at t=T (FEM | PINN | |err|) with probes ---
th = np.linspace(0, 2*np.pi, 100)
fv = fem_v(T); pv = pinn_v(T)
m = ~inside
fv[inside] = np.nan; pv[inside] = np.nan
relL2 = float(np.sqrt(np.nansum((pv-fv)**2)) / np.sqrt(np.nansum(fv[m]**2)))
for k, (fld, ttl, cm, vlim) in enumerate([
        (fv, "FEM  v-vel  t=T", "RdBu_r", 0.6),
        (pv, "inverse PINN  v-vel  t=T  relL2=%.3f" % relL2, "RdBu_r", 0.6),
        (np.abs(pv-fv), "|PINN-FEM|", "magma", 0.3)]):
    ax = fig.add_subplot(gs[1, k])
    im = ax.pcolormesh(gx, gy, fld, shading="auto", cmap=cm,
                       vmin=(-vlim if cm == "RdBu_r" else 0), vmax=vlim)
    ax.fill(0.5*np.cos(th), 0.5*np.sin(th), color="k")
    ax.plot(probe_xy[:, 0], probe_xy[:, 1], 'gP', ms=7, mec='k', mew=0.6,
            label='probes' if k == 0 else None)
    ax.set_aspect("equal"); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    ax.set_title(ttl, fontsize=9)
    if k == 0: ax.legend(fontsize=8, loc='upper right')
    fig.colorbar(im, ax=ax, shrink=0.7)

fig.suptitle("Inverse PINN inv%s (32x3, %d probes, nu0=%.3g): nu_rec=%.4f vs nu_true=%.4f  "
             "(rel err %.1f%%, Re_rec=%.1f)"
             % (TAG, probe_xy.shape[0], NU0, NU_REC, NU_TRUE,
                100*abs(NU_REC-NU_TRUE)/NU_TRUE, 1.0/NU_REC), fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("figures/pinn_inv%s_result.png" % TAG, dpi=120)
print("saved figures/pinn_inv%s_result.png" % TAG, flush=True)
