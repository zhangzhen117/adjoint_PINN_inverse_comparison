"""Post-process the forward PINN: load the trained model, evaluate on a grid,
compare against the FEM reference (validate.npz) across the shedding period,
save the predicted fields, and plot FEM vs PINN vs |error|.

Reuses the trained weights in history/pinn_fwd_model.pt -- no retraining.
"""
import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from cylinder_solver import CylinderConfig, create_solver

torch.set_default_dtype(torch.float64)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
X0, X1, Y0, Y1, R = -3.0, 10.0, -3.0, 3.0, 0.5

# ---- match training network exactly ----
class Net(nn.Module):
    def __init__(self, h=32, L=3, T=6.0):
        super().__init__()
        self.T = T
        layers = [nn.Linear(3, h), nn.Tanh()]
        for _ in range(L-1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers += [nn.Linear(h, 2)]
        self.net = nn.Sequential(*layers)
    def raw(self, x, y, t):
        xs = (x - X0)/(X1 - X0)*2 - 1; ys = y/Y1; ts = t/self.T*2 - 1
        out = self.net(torch.stack([xs, ys, ts], dim=-1))
        return out[..., 0], out[..., 1]

def velocity(model, x, y, t):
    psi, p = model.raw(x, y, t)
    u =  grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
    v = -grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    return u, v

# ---- load run metadata + model (TAG selects v1='' or v2='2' etc.) ----
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
meta = np.load(f'history/pinn_fwd{TAG}.npz')
T, t0a = float(meta['T']), float(meta['t0a'])
H = int(meta['hidden']) if 'hidden' in meta else 32
L = int(meta['layers']) if 'layers' in meta else 3
model = Net(h=H, L=L, T=T).to(DEV)
model.load_state_dict(torch.load(f'history/pinn_fwd{TAG}_model.pt', map_location=DEV,
                                 weights_only=True))
model.eval()
print("loaded model; T=%.4f t0a=%.3f device=%s" % (T, t0a, DEV), flush=True)

# ---- FEM mesh + snapshots ----
cfg = CylinderConfig(h_cyl=0.04, h_wake=0.08, h_far=0.5)
mesh, _, _ = create_solver(cfg)
nps = mesh.n_p2; XY = mesh.coords_p2
zt = np.load('history/validate.npz', mmap_mode='r'); times = np.array(zt['times'])

# ---- evaluation grid (cylinder masked) ----
nx, ny = 320, 150
gx = np.linspace(X0, X1, nx); gy = np.linspace(Y0, Y1, ny)
GX, GY = np.meshgrid(gx, gy)
inside = np.hypot(GX, GY) <= R + 1e-3
pts = np.column_stack([GX.ravel(), GY.ravel()])

def pinn_uv(tt):
    x = torch.tensor(pts[:, 0], device=DEV, requires_grad=True)
    y = torch.tensor(pts[:, 1], device=DEV, requires_grad=True)
    t = torch.tensor(np.full(len(pts), tt), device=DEV, requires_grad=True)
    u, v = velocity(model, x, y, t)
    return (u.detach().cpu().numpy().reshape(ny, nx),
            v.detach().cpu().numpy().reshape(ny, nx))

def fem_uv(idx):
    snap = np.array(zt['snaps'][idx])
    uv = np.column_stack([snap[:nps], snap[nps:2*nps]])
    li = LinearNDInterpolator(XY, uv); ne = NearestNDInterpolator(XY, uv)
    w = li(pts); bad = np.isnan(w[:, 0])
    if bad.any(): w[bad] = ne(pts[bad])
    return w[:, 0].reshape(ny, nx), w[:, 1].reshape(ny, nx)

# ---- compare across the period ----
fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
saved = {'gx': gx, 'gy': gy, 'fracs': np.array(fracs), 'T': T, 't0a': t0a}
errs = []
fig, axes = plt.subplots(len(fracs), 3, figsize=(16, 2.6*len(fracs)))
th = np.linspace(0, 2*np.pi, 100)
for i, fr in enumerate(fracs):
    tt = fr*T
    idx = int(np.argmin(np.abs(times - (t0a + tt))))
    fu, fv = fem_uv(idx)
    pu, pv = pinn_uv(tt)
    fu[inside.reshape(ny,nx)] = np.nan; fv[inside.reshape(ny,nx)] = np.nan
    pu[inside.reshape(ny,nx)] = np.nan; pv[inside.reshape(ny,nx)] = np.nan
    m = ~inside.reshape(ny, nx)
    e = np.sqrt(np.nansum((pu-fu)**2 + (pv-fv)**2)) / np.sqrt(np.nansum(fu[m]**2 + fv[m]**2))
    errs.append(e)
    saved[f'fem_v_{i}'] = fv; saved[f'pinn_v_{i}'] = pv
    saved[f'fem_u_{i}'] = fu; saved[f'pinn_u_{i}'] = pu
    for ax, fld, ttl in [(axes[i,0], fv, "FEM"), (axes[i,1], pv, "PINN"),
                         (axes[i,2], np.abs(pv-fv), "|PINN-FEM|")]:
        vlim = 0.6 if "PINN-FEM" not in ttl else 0.3
        cm = "RdBu_r" if "PINN-FEM" not in ttl else "magma"
        im = ax.pcolormesh(gx, gy, fld, shading="auto", cmap=cm,
                           vmin=(-vlim if cm=="RdBu_r" else 0), vmax=vlim)
        ax.fill(0.5*np.cos(th), 0.5*np.sin(th), color="k")
        ax.set_aspect("equal"); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
        ax.set_title(f"{ttl}  t={tt:.2f} ({fr*100:.0f}%)  v-vel" +
                     (f"  relL2={e:.3f}" if ttl=="PINN" else ""), fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    print("  t=%.3f (%.0f%%):  rel L2 = %.4e" % (tt, fr*100, e), flush=True)

fig.tight_layout()
fig.savefig(f"figures/pinn_fwd{TAG}_compare.png", dpi=120)
print(f"saved figures/pinn_fwd{TAG}_compare.png", flush=True)

saved['errs'] = np.array(errs)
np.savez(f"history/pinn_fwd{TAG}_fields.npz", **saved)
print(f"saved history/pinn_fwd{TAG}_fields.npz", flush=True)

# error-vs-time curve
plt.figure(figsize=(6,4))
plt.plot([f*100 for f in fracs], errs, 'o-')
plt.xlabel("% of shedding period"); plt.ylabel("rel L2 (velocity) vs FEM")
plt.title("Forward PINN drift over one period (3x32, T=%.2f)" % T)
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"figures/pinn_fwd{TAG}_error.png", dpi=120)
print(f"saved figures/pinn_fwd{TAG}_error.png", flush=True)
