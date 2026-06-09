"""
2D unsteady incompressible Navier-Stokes past a cylinder (Re=60), P2/P1
Taylor-Hood FEM with semi-implicit IMEX-SBDF2 time stepping: implicit BDF2
diffusion + monolithically-coupled pressure (a single sparse-LU saddle solve
per step, factored once per dt and reused), with 2nd-order explicit-extrapolated
convection (no Newton -- each step is one linear solve). The first step and any
adaptive-dt change use a one-step SBDF1 (implicit/explicit-Euler) bootstrap.

Purpose: forward solver + (later) discrete adjoint for scalar viscosity
identification (theta = log nu), compared against a PINN. See
methodology_Cylinder.md.

The element-level assembly mirrors the 2D_cavity FEMAssembler (7-pt Gauss,
P2/P1 Lagrange, vectorized Jacobians + COO scatter); it depends only on the
triangulation, so it is reused here on the gmsh unstructured mesh.
"""

import numpy as np
from scipy.sparse import csr_matrix, bmat, diags
from scipy.sparse.linalg import spsolve, splu
from dataclasses import dataclass

from cylinder_mesh import CylinderMesh


# =============================================================================
# Config
# =============================================================================
@dataclass
class CylinderConfig:
    Re: float = 60.0
    nu: float = None
    U_inf: float = 1.0
    D: float = 1.0
    # mesh
    x0: float = -3.0; x1: float = 10.0; y0: float = -3.0; y1: float = 3.0
    h_cyl: float = 0.035; h_wake: float = 0.08; h_far: float = 0.4
    # time
    dt: float = 0.02
    eps_p: float = 1e-10   # pressure safeguard (outlet do-nothing sets level)

    def __post_init__(self):
        if self.nu is None:
            self.nu = self.U_inf * self.D / self.Re


# =============================================================================
# FEM assembler (P2/P1) -- reused design from cavity_solver.FEMAssembler
# =============================================================================
class FEMAssembler:
    def __init__(self, mesh: CylinderMesh):
        self.mesh = mesh
        self._setup_quadrature()
        self._precompute_basis()
        self._precompute_geometry()

    def _setup_quadrature(self):
        self.qpts = np.array([
            [1/3, 1/3],
            [0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.059715871789770],
            [0.470142064105115, 0.470142064105115],
            [0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.797426985353087],
            [0.101286507323456, 0.101286507323456],
        ])
        self.qwts = np.array([0.225,
                              0.132394152788506, 0.132394152788506, 0.132394152788506,
                              0.125939180544827, 0.125939180544827, 0.125939180544827]) / 2

    def _precompute_basis(self):
        nq = len(self.qwts)
        self.phi_p2 = np.zeros((nq, 6)); self.dphi_p2 = np.zeros((nq, 6, 2))
        for q, (xi, eta) in enumerate(self.qpts):
            lam = [1 - xi - eta, xi, eta]
            self.phi_p2[q, 0] = lam[0]*(2*lam[0]-1)
            self.phi_p2[q, 1] = lam[1]*(2*lam[1]-1)
            self.phi_p2[q, 2] = lam[2]*(2*lam[2]-1)
            self.phi_p2[q, 3] = 4*lam[0]*lam[1]
            self.phi_p2[q, 4] = 4*lam[1]*lam[2]
            self.phi_p2[q, 5] = 4*lam[0]*lam[2]
            dlam = [[-1, -1], [1, 0], [0, 1]]
            for i in range(3):
                self.dphi_p2[q, i] = (4*lam[i]-1)*np.array(dlam[i])
            self.dphi_p2[q, 3] = 4*(lam[1]*np.array(dlam[0]) + lam[0]*np.array(dlam[1]))
            self.dphi_p2[q, 4] = 4*(lam[2]*np.array(dlam[1]) + lam[1]*np.array(dlam[2]))
            self.dphi_p2[q, 5] = 4*(lam[2]*np.array(dlam[0]) + lam[0]*np.array(dlam[2]))
        self.phi_p1 = np.zeros((nq, 3)); self.dphi_p1 = np.zeros((nq, 3, 2))
        for q, (xi, eta) in enumerate(self.qpts):
            self.phi_p1[q] = [1-xi-eta, xi, eta]
            self.dphi_p1[q] = [[-1, -1], [1, 0], [0, 1]]

    def _precompute_geometry(self):
        mesh = self.mesh
        verts = mesh.coords_p2[mesh.elem_p2[:, :3]]          # (nE,3,2)
        dx1 = verts[:, 1] - verts[:, 0]
        dx2 = verts[:, 2] - verts[:, 0]
        det = dx1[:, 0]*dx2[:, 1] - dx1[:, 1]*dx2[:, 0]
        self._detJ = np.abs(det)
        inv_det = 1.0/det
        self._Jinv = np.zeros((mesh.n_elem, 2, 2))
        self._Jinv[:, 0, 0] = dx2[:, 1]*inv_det
        self._Jinv[:, 0, 1] = -dx2[:, 0]*inv_det
        self._Jinv[:, 1, 0] = -dx1[:, 1]*inv_det
        self._Jinv[:, 1, 1] = dx1[:, 0]*inv_det
        self._dphi_phys = np.einsum('qik,ekd->eqid', self.dphi_p2, self._Jinv)
        self._wdetJ = self._detJ[:, None]*self.qwts[None, :]
        e2 = mesh.elem_p2
        self._coo_rows_p2 = np.repeat(e2[:, :, None], 6, axis=2).ravel()
        self._coo_cols_p2 = np.repeat(e2[:, None, :], 6, axis=1).ravel()
        e1 = mesh.elem_p1
        self._coo_rows_p1 = np.repeat(e1[:, :, None], 3, axis=2).ravel()
        self._coo_cols_p1 = np.repeat(e1[:, None, :], 3, axis=1).ravel()
        self._div_rows = np.repeat(e1[:, :, None], 6, axis=2).ravel()
        self._div_cols = np.repeat(e2[:, None, :], 3, axis=1).ravel()

    def assemble_mass_p2(self):
        n = self.mesh.n_p2
        phi_outer = np.einsum('qi,qj->qij', self.phi_p2, self.phi_p2)
        Me = np.einsum('eq,qij->eij', self._wdetJ, phi_outer)
        return csr_matrix((Me.ravel(), (self._coo_rows_p2, self._coo_cols_p2)), shape=(n, n))

    def assemble_mass_p1(self):
        n = self.mesh.n_p1
        phi_outer = np.einsum('qi,qj->qij', self.phi_p1, self.phi_p1)
        Me = np.einsum('eq,qij->eij', self._wdetJ, phi_outer)
        return csr_matrix((Me.ravel(), (self._coo_rows_p1, self._coo_cols_p1)), shape=(n, n))

    def assemble_stiffness_p2(self):
        n = self.mesh.n_p2
        Ke = np.einsum('eq,eqid,eqjd->eij', self._wdetJ, self._dphi_phys, self._dphi_phys)
        return csr_matrix((Ke.ravel(), (self._coo_rows_p2, self._coo_cols_p2)), shape=(n, n))

    def assemble_divergence(self):
        n_p1, n_p2 = self.mesh.n_p1, self.mesh.n_p2
        Bx = -np.einsum('eq,qi,eqj->eij', self._wdetJ, self.phi_p1, self._dphi_phys[:, :, :, 0])
        By = -np.einsum('eq,qi,eqj->eij', self._wdetJ, self.phi_p1, self._dphi_phys[:, :, :, 1])
        data = np.concatenate([Bx.ravel(), By.ravel()])
        rows = np.concatenate([self._div_rows, self._div_rows])
        cols = np.concatenate([self._div_cols, self._div_cols + n_p2])
        return csr_matrix((data, (rows, cols)), shape=(n_p1, 2*n_p2))

    def assemble_convection(self, u):
        n_p2 = self.mesh.n_p2
        ux, uy = u[:n_p2], u[n_p2:]
        elem = self.mesh.elem_p2
        ux_q = np.einsum('qj,ej->eq', self.phi_p2, ux[elem])
        uy_q = np.einsum('qj,ej->eq', self.phi_p2, uy[elem])
        u_grad_phi = (ux_q[:, :, None]*self._dphi_phys[:, :, :, 0] +
                      uy_q[:, :, None]*self._dphi_phys[:, :, :, 1])
        Ce = np.einsum('eq,qi,eqj->eij', self._wdetJ, self.phi_p2, u_grad_phi)
        Cs = csr_matrix((Ce.ravel(), (self._coo_rows_p2, self._coo_cols_p2)), shape=(n_p2, n_p2))
        return bmat([[Cs, None], [None, Cs]], format='csr')

    def assemble_convection_jacobian(self, u):
        n_p2 = self.mesh.n_p2
        ux, uy = u[:n_p2], u[n_p2:]
        elem = self.mesh.elem_p2
        grad_ux = np.einsum('eqid,ei->eqd', self._dphi_phys, ux[elem])
        grad_uy = np.einsum('eqid,ei->eqd', self._dphi_phys, uy[elem])
        phi_outer = np.einsum('qi,qj->qij', self.phi_p2, self.phi_p2)
        Jxx = np.einsum('eq,qij,eq->eij', self._wdetJ, phi_outer, grad_ux[:, :, 0])
        Jxy = np.einsum('eq,qij,eq->eij', self._wdetJ, phi_outer, grad_ux[:, :, 1])
        Jyx = np.einsum('eq,qij,eq->eij', self._wdetJ, phi_outer, grad_uy[:, :, 0])
        Jyy = np.einsum('eq,qij,eq->eij', self._wdetJ, phi_outer, grad_uy[:, :, 1])
        r, c = self._coo_rows_p2, self._coo_cols_p2
        S = lambda d: csr_matrix((d.ravel(), (r, c)), shape=(n_p2, n_p2))
        return bmat([[S(Jxx), S(Jxy)], [S(Jyx), S(Jyy)]], format='csr')


# =============================================================================
# Unsteady NS solver (semi-implicit IMEX-SBDF2; one linear saddle solve / step)
# =============================================================================
class CylinderSolver:
    def __init__(self, mesh: CylinderMesh, asm: FEMAssembler, cfg: CylinderConfig):
        self.mesh, self.asm, self.cfg = mesh, asm, cfg
        self.M = asm.assemble_mass_p2()
        self.K = asm.assemble_stiffness_p2()
        self.B = asm.assemble_divergence()
        self.M_u = bmat([[self.M, None], [None, self.M]], format='csr')
        self.K_u = bmat([[self.K, None], [None, self.K]], format='csr')
        self.M_p = asm.assemble_mass_p1()

        # Dirichlet masks (velocity)
        bnd = mesh.bnd_u
        free = np.ones(mesh.n_u); free[bnd] = 0.0
        self._D_u = diags(free).tocsr()
        self._I_bnd = diags(1.0 - free).tocsr()
        self._neg_eps_Mp = (-cfg.eps_p * self.M_p).tocsr()
        self._B_bc = (self.B @ self._D_u).tocsr()
        self._BT_bc = self._B_bc.T.tocsr()
        self.g = mesh.u_dirichlet     # prescribed velocity (lift)

        # per-element length scale (sqrt of triangle area) for CFL estimates
        self._h_elem = np.sqrt(self.asm._detJ / 2.0)
        self._tri = None              # lazily-built triangulation for plotting

    # ---- one-shot saddle-point solve (homogeneous on bnd); generic helper,
    # NOT used by the SBDF2 stepping path (which factors once via _make_stepper) ----
    def _solve_saddle(self, A, rhs_u, rhs_p):
        A_bc = (self._D_u @ A @ self._D_u + self._I_bnd).tocsr()
        rhs_u_bc = self._D_u @ rhs_u
        Kf = bmat([[A_bc, self._BT_bc], [self._B_bc, self._neg_eps_Mp]], format='csc')
        sol = spsolve(Kf, np.concatenate([rhs_u_bc, rhs_p]))
        return sol[:self.mesh.n_u], sol[self.mesh.n_u:]

    # ------------------------------------------------------------------------
    # Semi-implicit IMEX time stepping: implicit diffusion + pressure,
    # explicit (extrapolated) convection on the RHS. The LHS saddle operator
    #   [ alpha*M_u + nu*K_u ,  B^T ;  B , -eps*M_p ]
    # is CONSTANT in time -> factor ONCE (splu) and back-substitute each step.
    # CFL-limited (dt <~ h/|u|) but no Newton, no per-step refactorization.
    # ------------------------------------------------------------------------
    def _make_stepper(self, alpha, nu):
        """Build + factor the constant BC-reduced saddle operator for a given
        BDF time coefficient alpha (1/dt for BDF1, 3/(2dt) for BDF2)."""
        A_u = (alpha*self.M_u + nu*self.K_u).tocsr()
        A_bc = (self._D_u @ A_u @ self._D_u + self._I_bnd).tocsr()
        Kf = bmat([[A_bc, self._BT_bc], [self._B_bc, self._neg_eps_Mp]], format='csc')
        u0 = self.g                       # Dirichlet lift (values on bnd, 0 elsewhere)
        return dict(lu=splu(Kf), alpha=alpha,
                    lift_u=A_u @ u0,      # constant momentum-RHS reduction
                    lift_p=self.B @ u0)   # constant continuity-RHS reduction

    def _apply_stepper(self, st, b_u, r=1.0, b_p=None):
        """One back-substitution. Solves
            [A_u, B^T; B, -eps Mp] [u; p] = [b_u; b_p]
        with the (possibly ramped) Dirichlet lift r*g handled by static
        condensation (u = r*g + u~, u~ = 0 on Dirichlet DOFs)."""
        n_u = self.mesh.n_u
        rhs_u = self._D_u @ (b_u - r*st['lift_u'])
        rhs_p = (-(r*st['lift_p']) if b_p is None else b_p - r*st['lift_p'])
        sol = st['lu'].solve(np.concatenate([rhs_u, rhs_p]))
        return r*self.g + sol[:n_u], sol[n_u:]

    def _conv_load(self, u):
        """Convection load vector N(u) = (u.grad)u tested against P2 = C(u) @ u."""
        return self.asm.assemble_convection(u) @ u

    def _ramp(self, t, ramp_T):
        """Smooth C1 inlet ramp from 0 to 1 over [0, ramp_T] (0 -> impulsive)."""
        if ramp_T <= 0:
            return 1.0
        s = min(t/ramp_T, 1.0)
        return 0.5*(1.0 - np.cos(np.pi*s))

    def _blob(self, center, sigma, amp):
        """Off-axis Gaussian perturbation added to the vertical velocity (uy),
        used as a one-time symmetry-breaking IC blob to seed vortex shedding.
        Returns a full velocity-DOF vector (zero in ux, Gaussian in uy, zeroed
        on Dirichlet uy nodes)."""
        xc, yc = center
        x = self.mesh.coords_p2[:, 0]; y = self.mesh.coords_p2[:, 1]
        g = amp*np.exp(-((x - xc)**2 + (y - yc)**2)/(2.0*sigma**2))
        v = np.zeros(self.mesh.n_u)
        v[self.mesh.n_p2:] = g
        v[self.mesh.bnd_u] = 0.0          # never perturb Dirichlet DOFs
        return v

    def _cfl_rate(self, u):
        """max_element |u| / h_eff, with P2 effective spacing h_eff = h_elem/2.
        The convective CFL of a step of size dt is dt * _cfl_rate(u)."""
        npx = self.mesh.n_p2
        speed = np.sqrt(u[:npx]**2 + u[npx:]**2)
        spe_elem = speed[self.mesh.elem_p2].max(axis=1)        # (nE,)
        return float((spe_elem / (0.5*self._h_elem)).max())

    def _save_frame(self, u, t, fname, cd=None, cl=None):
        """Render the vertical-velocity field (shows the wake) to a PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        if self._tri is None:
            self._tri = mtri.Triangulation(self.mesh.coords_p1[:, 0],
                                           self.mesh.coords_p1[:, 1],
                                           self.mesh.elem_p1)
        uy_p1 = u[self.mesh.n_p2 + self.mesh._p1_to_p2]
        fig, ax = plt.subplots(figsize=(11, 5))
        tc = ax.tripcolor(self._tri, uy_p1, shading="gouraud", cmap="RdBu_r",
                          vmin=-0.6, vmax=0.6)
        th = np.linspace(0, 2*np.pi, 100)
        ax.fill(0.5*np.cos(th), 0.5*np.sin(th), color="k")
        ax.set_aspect("equal"); ax.set_xlim(self.mesh.x0, self.mesh.x1)
        ax.set_ylim(self.mesh.y0, self.mesh.y1)
        ttl = f"v at t={t:.1f}"
        if cd is not None:
            ttl += f"   C_D={cd:.3f}  C_L={cl:.3f}"
        ax.set_title(ttl); fig.colorbar(tc, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)

    def solve_forward(self, T, nu=None, u0=None, dt=None, t0=0.0, ramp_T=2.0,
                      store_all=False, save_dt=None, verbose=False,
                      blob_t=None, blob_amp=0.1, blob_center=(1.0, 0.3),
                      blob_sigma=0.3,
                      adaptive=False, cfl_target=0.9, dt_min=2e-3, dt_max=5e-2,
                      plot_every_t=None, plot_prefix="figures/frame"):
        """Integrate to time T using IMEX SBDF2 (implicit Euler bootstrap).

        Convection is explicit via 2nd-order extrapolation N* = 2N(u^n)-N(u^{n-1}).
        Inlet ramps 0->U_inf over [t0, t0+ramp_T] (C1); flow starts from rest
        (u0=None). If blob_t is set, a one-time off-axis Gaussian uy blob seeds
        shedding at the first step with t >= blob_t.

        Adaptive time step (adaptive=True): each step dt is chosen so the
        convective CFL ~ cfl_target (~0.9). To preserve the factor-once speed
        (the LHS depends on dt), dt is only *changed* — and the LU refactored
        (cached per dt) with a one-step BDF1 restart — when the CFL drifts
        outside a hysteresis band; otherwise the cached factorization is reused.

        Sampling/plotting are time-based: a snapshot every save_dt time units;
        a wake PNG every plot_every_t time units (if set). store_all keeps every
        step (for adjoint checkpointing)."""
        nu = nu or self.cfg.nu
        dt = dt or self.cfg.dt
        if adaptive:
            dt = min(dt_max, max(dt_min, dt))
        save_dt = save_dt if save_dt is not None else dt

        # stepper factorization cache, keyed by rounded dt
        cache = {}
        def stepper(dt_val, bdf1):
            key = (round(dt_val, 9), bdf1)
            if key not in cache:
                cache[key] = self._make_stepper((1.0 if bdf1 else 1.5)/dt_val, nu)
            return cache[key]

        blob = (self._blob(blob_center, blob_sigma, blob_amp)
                if blob_t is not None else None)
        blob_done = False

        u = (self._ramp(t0, ramp_T)*self.g if u0 is None else u0.copy())
        u_n = u.copy(); u_nm1 = u.copy()
        N_n = self._conv_load(u_n); N_nm1 = N_n.copy()

        times = [t0]; snaps = [u.copy()]; save_times = [t0]
        all_u = [u.copy()] if store_all else None
        CD, CL, t_force, dts = [], [], [], []

        t = t0; step = 0; need_restart = True
        next_save = t0 + save_dt
        next_plot = (t0 + plot_every_t) if plot_every_t else np.inf
        n_refactor = 0
        while t < T - 1e-9:
            # ---- adaptive dt selection (hysteresis around cfl_target) ----
            if adaptive:
                rate = self._cfl_rate(u_n)
                cur_cfl = dt*rate
                if cur_cfl > cfl_target*1.05 or cur_cfl < cfl_target*0.5:
                    new_dt = min(dt_max, max(dt_min, cfl_target/max(rate, 1e-12)))
                    if abs(new_dt - dt)/dt > 1e-2:
                        dt = new_dt; need_restart = True; n_refactor += 1
            if t + dt > T:                      # last partial step
                dt = T - t; need_restart = True
            t_next = t + dt
            r = self._ramp(t_next, ramp_T)

            if need_restart:                    # IMEX Euler (re-establishes spacing)
                st = stepper(dt, bdf1=True)
                b_u = self.M_u @ (u_n/dt) - N_n
                u_new, p_new = self._apply_stepper(st, b_u, r=r)
            else:                               # SBDF2
                st = stepper(dt, bdf1=False)
                b_u = self.M_u @ ((4.0*u_n - u_nm1)/(2.0*dt)) - (2.0*N_n - N_nm1)
                u_new, p_new = self._apply_stepper(st, b_u, r=r)
            need_restart = False

            if blob is not None and not blob_done and t_next >= blob_t:
                u_new = u_new + blob
                blob_done = True
                if verbose:
                    print(f"  [blob injected at t={t_next:.3f}, amp={blob_amp}]", flush=True)

            # assemble C(u_new) ONCE: shared by the force residual and next N(u)
            Cu_new = self.asm.assemble_convection(u_new) @ u_new
            cd, cl = self._cylinder_force(u_new, p_new, u_n, u_nm1, dt,
                                          nu, 1 if step == 0 else 2, Cu=Cu_new)
            CD.append(cd); CL.append(cl); t_force.append(t_next); dts.append(dt)

            u_nm1 = u_n; u_n = u_new
            N_nm1 = N_n; N_n = Cu_new
            t = t_next; step += 1

            if store_all:
                all_u.append(u_new.copy())
            if t >= next_save - 1e-12:
                snaps.append(u_new.copy()); save_times.append(t); next_save += save_dt
            if t >= next_plot - 1e-12:
                self._save_frame(u_new, t, f"{plot_prefix}_{step:06d}.png", cd, cl)
                next_plot += plot_every_t
            if verbose and step % 200 == 0:
                print(f"  step {step} t={t:.3f} dt={dt:.4f} CFL={dt*self._cfl_rate(u_new):.2f} "
                      f"|u|max={np.abs(u_new).max():.3f} CD={cd:.3f} CL={cl:.3f}", flush=True)

        if verbose and adaptive:
            print(f"  [adaptive: {n_refactor} dt-changes/refactorizations, "
                  f"final dt={dt:.4f}]", flush=True)
        out = dict(times=np.array(save_times), snaps=np.array(snaps),
                   u_final=u_n, u_prev=u_nm1, p_final=p_new,
                   CD=np.array(CD), CL=np.array(CL), t_force=np.array(t_force),
                   dts=np.array(dts), nu=nu, n_steps=step)
        if store_all:
            out['all_u'] = np.array(all_u)
        return out

    def _cylinder_force(self, u, p, u_n, u_nm1, dt, nu, step, Cu=None):
        """Drag/lift on cylinder by the consistent reaction (residual) method.
        The momentum-equation residual (no BC mask) summed over the cylinder
        test functions equals minus the boundary traction integral, so the
        force the fluid exerts on the body is -r. C_D = 2 F_x, C_L = 2 F_y
        (rho = U = D = 1).

        Cu = C(u) @ u, the convection load, may be passed in to avoid
        re-assembling the convection operator (the time stepper already needs
        it for the next step's N(u)); if None it is assembled here."""
        if step == 1:
            u_t = (u - u_n)/dt
        else:
            u_t = (1.5*u - 2.0*u_n + 0.5*u_nm1)/dt
        if Cu is None:
            Cu = self.asm.assemble_convection(u) @ u
        r = self.M_u @ u_t + Cu + nu*self.K_u @ u + self.B.T @ p
        npx = self.mesh.n_p2
        cyl = self.mesh.bnd["cylinder"]
        Fx = -r[cyl].sum()
        Fy = -r[npx + cyl].sum()
        return 2.0*Fx, 2.0*Fy


# =============================================================================
# Discrete adjoint for scalar viscosity identification  (theta = log nu)
# =============================================================================
class CylinderAdjoint:
    """Discrete adjoint of the IMEX SBDF2 forward map for the scalar control
    theta = log(nu).

    Objective (single full-field velocity observation at step K):
        J(theta) = 0.5*w*(u^K - u_obs)^T M_u (u^K - u_obs)
                 + 0.5*alpha*(theta - theta0)^2.

    Key structural fact: with explicit convection the implicit step operator
    G = [[alpha*M_u + nu*K_u, B^T],[B, -eps*M_p]] is SYMMETRIC (no convection on
    the LHS), so the adjoint step solves G^T = G with the SAME splu factor.
    The (nonsymmetric) convection coupling enters only as transposed mat-vecs
    J_N(u^n)^T lambda with J_N(u) = C(u) + C_u(u), assembled from stored states.
    The forward states u^0..u^K are checkpointed (store_all=True)."""

    def __init__(self, solver: 'CylinderSolver'):
        self.ns = solver
        self.mesh = solver.mesh
        self.asm = solver.asm

    def _adjoint_solve(self, st, rhs_u):
        """Solve the symmetric BC-reduced saddle system for the adjoint
        (homogeneous Dirichlet: lambda_u = 0 on Dirichlet DOFs)."""
        ns = self.ns
        n_u, n_p = self.mesh.n_u, self.mesh.n_p
        ru = ns._D_u @ rhs_u
        sol = st['lu'].solve(np.concatenate([ru, np.zeros(n_p)]))
        return sol[:n_u], sol[n_u:]

    def value_and_grad(self, theta, u0, K, u_obs, dt, w=1.0,
                       alpha=0.0, theta0=0.0):
        """Return (J, dJ/dtheta, info). Runs the forward inverse model from the
        snapshot u0 for K fixed-dt steps (BDF1 bootstrap + SBDF2), then the
        backward adjoint sweep. nu = exp(theta)."""
        ns = self.ns
        nu = float(np.exp(theta))
        M_u, K_u = ns.M_u, ns.K_u

        # ---- forward (fixed dt, full BC, no ramp/blob, checkpoint all) ----
        out = ns.solve_forward(T=K*dt, nu=nu, u0=u0, dt=dt, ramp_T=0.0,
                               adaptive=False, store_all=True, save_dt=K*dt)
        U = out['all_u']                       # (K+1, n_u): U[n] = u^n
        uK = U[K]
        misfit = uK - u_obs
        J = 0.5*w*(misfit @ (M_u @ misfit)) + 0.5*alpha*(theta - theta0)**2

        # step operators (G symmetric -> reuse for adjoint); BDF1 for n=1
        st1 = ns._make_stepper(1.0/dt, nu)
        st2 = ns._make_stepper(1.5/dt, nu)

        # ---- backward adjoint sweep ----
        lam_np1 = np.zeros(ns.mesh.n_u)        # lambda_u^{n+1}
        lam_np2 = np.zeros(ns.mesh.n_u)        # lambda_u^{n+2}
        grad_nu = 0.0
        for n in range(K, 0, -1):
            u_n = U[n]
            rhs_u = np.zeros(ns.mesh.n_u)
            if n == K:
                rhs_u -= w*(M_u @ misfit)                      # -dJ/du^K
            if n < K:                                          # convection coupling
                JN = self.asm.assemble_convection(u_n) + \
                     self.asm.assemble_convection_jacobian(u_n)
                JNT = JN.T.tocsr()
                if n + 1 <= K:
                    rhs_u += (2.0/dt)*(M_u @ lam_np1) - 2.0*(JNT @ lam_np1)
                if n + 2 <= K:
                    rhs_u += -(1.0/(2.0*dt))*(M_u @ lam_np2) + (JNT @ lam_np2)
            st = st1 if n == 1 else st2
            lam_u, _ = self._adjoint_solve(st, rhs_u)
            grad_nu += lam_u @ (K_u @ u_n)                     # lambda^n . dR^n/dnu
            lam_np2 = lam_np1; lam_np1 = lam_u

        grad_theta = nu*grad_nu + alpha*(theta - theta0)
        info = dict(nu=nu, J=J, misfit_norm=float(np.sqrt(misfit @ (M_u @ misfit))),
                    u_pred=uK)
        return J, grad_theta, info

    def value_and_grad_obs(self, theta, u0, obs_steps, obs_data, P, dt,
                           w=1.0, alpha=0.0, theta0=0.0):
        """Generalized objective: SPARSE-in-space (probe operator P) and
        MULTI-TIME observations. P is a (2*n_probe, n_u) sparse point-evaluation
        operator (rows = ux at probes, then uy at probes); obs_steps lists the
        step indices n_k in 1..K at which P u^{n_k} is observed; obs_data[k] is
        the corresponding target vector P u^{n_k}(nu_true).

            J(theta) = 0.5 * sum_k w_k ||P u^{n_k} - d^k||^2 + 0.5*alpha*(theta-theta0)^2

        The misfit is plain Euclidean on the sampled values (point observations
        carry no FE mass weight, unlike the full-field M_u norm). Returns
        (J, dJ/dtheta, info). nu = exp(theta). Structure of the backward sweep
        is identical to value_and_grad; only the adjoint source changes -- it is
        injected as -w_k P^T (P u^{n_k} - d^k) at each observation step."""
        ns = self.ns
        nu = float(np.exp(theta))
        M_u, K_u = ns.M_u, ns.K_u
        obs_steps = [int(s) for s in obs_steps]
        K = max(obs_steps)
        ws = np.full(len(obs_steps), float(w)) if np.isscalar(w) else np.asarray(w, float)

        # ---- forward (fixed dt, full BC, no ramp/blob, checkpoint all) ----
        out = ns.solve_forward(T=K*dt, nu=nu, u0=u0, dt=dt, ramp_T=0.0,
                               adaptive=False, store_all=True, save_dt=K*dt)
        U = out['all_u']                       # (K+1, n_u): U[n] = u^n

        # misfit + per-step adjoint source  src[n] = -dJ/du^n
        J = 0.5*alpha*(theta - theta0)**2
        src = {}
        misfit2 = 0.0
        for k, n in enumerate(obs_steps):
            r = P @ U[n] - obs_data[k]
            J += 0.5*ws[k]*(r @ r)
            misfit2 += r @ r
            src[n] = -ws[k]*(P.T @ r)

        # step operators (G symmetric -> reuse for adjoint); BDF1 for n=1
        st1 = ns._make_stepper(1.0/dt, nu)
        st2 = ns._make_stepper(1.5/dt, nu)

        # ---- backward adjoint sweep ----
        lam_np1 = np.zeros(ns.mesh.n_u)
        lam_np2 = np.zeros(ns.mesh.n_u)
        grad_nu = 0.0
        for n in range(K, 0, -1):
            u_n = U[n]
            rhs_u = src[n].copy() if n in src else np.zeros(ns.mesh.n_u)
            if n < K:                                          # convection coupling
                JN = self.asm.assemble_convection(u_n) + \
                     self.asm.assemble_convection_jacobian(u_n)
                JNT = JN.T.tocsr()
                if n + 1 <= K:
                    rhs_u += (2.0/dt)*(M_u @ lam_np1) - 2.0*(JNT @ lam_np1)
                if n + 2 <= K:
                    rhs_u += -(1.0/(2.0*dt))*(M_u @ lam_np2) + (JNT @ lam_np2)
            st = st1 if n == 1 else st2
            lam_u, _ = self._adjoint_solve(st, rhs_u)
            grad_nu += lam_u @ (K_u @ u_n)
            lam_np2 = lam_np1; lam_np1 = lam_u

        grad_theta = nu*grad_nu + alpha*(theta - theta0)
        info = dict(nu=nu, J=J, misfit_norm=float(np.sqrt(misfit2)))
        return J, grad_theta, info


def build_probe_operator(mesh, probe_xy):
    """Sparse (2*P, n_u) operator evaluating the P2 velocity field at P points.

    Row j      -> u_x at probe j;  row P+j -> u_y at probe j.
    Each probe is located in the triangle that contains it (barycentric test on
    the corner vertices), and the 6 quadratic P2 shape functions are evaluated
    there, so P u returns the FE-interpolated velocity at the probe locations.
    Probes are expected to lie strictly inside the fluid domain."""
    probe_xy = np.atleast_2d(np.asarray(probe_xy, float))
    coords, elem = mesh.coords_p2, mesh.elem_p2
    n_p2 = mesh.n_p2
    v = coords[elem[:, :3]]                       # (nE, 3, 2) corner verts
    v0 = v[:, 0]; d1 = v[:, 1] - v0; d2 = v[:, 2] - v0
    det = d1[:, 0]*d2[:, 1] - d1[:, 1]*d2[:, 0]
    P = len(probe_xy)
    rows, cols, data = [], [], []
    for j, (px, py) in enumerate(probe_xy):
        dx = px - v0[:, 0]; dy = py - v0[:, 1]
        xi  = ( d2[:, 1]*dx - d2[:, 0]*dy)/det     # solve [d1 d2][xi;eta]=(dx,dy)
        eta = (-d1[:, 1]*dx + d1[:, 0]*dy)/det
        l0, l1, l2 = 1.0 - xi - eta, xi, eta
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        e = int(np.argmax(inside)) if inside.any() else \
            int(np.argmax(np.minimum.reduce([l0, l1, l2])))
        a, b, c = l0[e], l1[e], l2[e]
        phi = np.array([a*(2*a-1), b*(2*b-1), c*(2*c-1), 4*a*b, 4*b*c, 4*a*c])
        nodes = elem[e]                            # 6 P2 node indices
        for i in range(6):
            rows.append(j);     cols.append(nodes[i]);          data.append(phi[i])
            rows.append(P + j); cols.append(n_p2 + nodes[i]);   data.append(phi[i])
    return csr_matrix((data, (rows, cols)), shape=(2*P, 2*n_p2))


def create_solver(cfg: CylinderConfig = None):
    cfg = cfg or CylinderConfig()
    mesh = CylinderMesh(cfg.x0, cfg.x1, cfg.y0, cfg.y1, cfg.D,
                        cfg.h_cyl, cfg.h_wake, cfg.h_far)
    asm = FEMAssembler(mesh)
    solver = CylinderSolver(mesh, asm, cfg)
    return mesh, asm, solver
