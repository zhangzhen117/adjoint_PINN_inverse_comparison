# darcy_inverse_fast.py
"""
Adjoint-based inverse solver for Darcy flow with sparse observations (FAST version).

Key speedups vs. baseline:
1) Vectorized global assembly using precomputed COO indices (no Python triple loops).
2) Interior-only solve (no "apply_bc" row/col editing each iteration).
3) Vectorized gradient via einsum over packed element tensors.

Inverse problem:
    Given sparse pressure observations y = P*u†, recover elementwise piecewise-constant
    log-permeability field m, where k_e = exp(m_e).

PDE:
    -∇·(k∇u) = f  in Ω = (0,1)²
     u = 0        on ∂Ω

Objective:
    J(m) = ½||P*u(m) - y_obs||² + β/2||m||² + γ/2 ∑_{edges} (m_i - m_j)^2 / h_ij

Adjoint:
    A(m)^T λ = P^T(Pu - y_obs)

Gradient (elementwise):
    ∂J/∂m_e = -k_e * (λ_e^T A_e^(0) u_e) + β m_e + (H1 terms if γ>0)

Notes:
- Requires forward mesh utilities:
    from darcy_solver import create_triangular_mesh, create_p2_mesh
  which should provide:
    create_triangular_mesh(nx, ny) -> (nodes_p1, elements_p1, boundary_nodes_p1)
    create_p2_mesh(nodes_p1, elements_p1) -> (nodes_p2, elements_p2, boundary_nodes_p2)

- For large problems, consider solver_type='amg' with pyamg installed.

Author: ChatGPT (optimized refactor)
"""

import time
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False

from darcy_solver import create_triangular_mesh, create_p2_mesh


# =============================================================================
# Observation operator
# =============================================================================

class ObservationOperator:
    """
    Observation operator y = P @ u_fem for sparse point observations.

    Builds a sparse interpolation matrix P using barycentric coordinates (P1) or
    quadratic P2 basis functions (P2) on the containing element.

    If elements is None, falls back to scipy's LinearNDInterpolator (slower).
    """

    def __init__(self, fem_nodes, obs_points, elements=None, element_degree=2):
        self.fem_nodes = np.asarray(fem_nodes)
        self.obs_points = np.asarray(obs_points)
        self.n_obs = self.obs_points.shape[0]
        self.n_fem = self.fem_nodes.shape[0]
        self.elements = elements
        self.element_degree = int(element_degree)

        self._build_interpolation_weights()

    def _build_interpolation_weights(self):
        if self.elements is None:
            self._use_scipy_interp = True
            self.P = None
            return

        self._use_scipy_interp = False
        P_data, P_rows, P_cols = [], [], []

        # For moderate meshes, a brute force element search at init is ok (one-time cost).
        for obs_idx, pt in enumerate(self.obs_points):
            found = False
            for elem_nodes in self.elements:
                coords = self.fem_nodes[np.asarray(elem_nodes[:3], dtype=np.int32)]

                v0 = coords[1] - coords[0]
                v1 = coords[2] - coords[0]
                v2 = pt - coords[0]

                d00 = np.dot(v0, v0)
                d01 = np.dot(v0, v1)
                d11 = np.dot(v1, v1)
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)

                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-14:
                    continue

                xi = (d11 * d20 - d01 * d21) / denom
                eta = (d00 * d21 - d01 * d20) / denom

                tol = 1e-10
                if (xi >= -tol) and (eta >= -tol) and (xi + eta <= 1 + tol):
                    found = True

                    if self.element_degree == 1:
                        lam0 = 1 - xi - eta
                        lam1 = xi
                        lam2 = eta
                        weights = [lam0, lam1, lam2]
                        local_nodes = elem_nodes[:3]
                    else:
                        lam0 = 1 - xi - eta
                        lam1 = xi
                        lam2 = eta
                        weights = [
                            lam0 * (2 * lam0 - 1),
                            lam1 * (2 * lam1 - 1),
                            lam2 * (2 * lam2 - 1),
                            4 * lam0 * lam1,
                            4 * lam1 * lam2,
                            4 * lam2 * lam0,
                        ]
                        local_nodes = elem_nodes

                    for w, node in zip(weights, local_nodes):
                        if abs(w) > 1e-14:
                            P_data.append(w)
                            P_rows.append(obs_idx)
                            P_cols.append(int(node))
                    break

            if not found:
                print(f"Warning: observation point {obs_idx} at {pt} not found in mesh")

        self.P = csr_matrix((P_data, (P_rows, P_cols)), shape=(self.n_obs, self.n_fem))

    def apply(self, u_fem):
        u_fem = np.asarray(u_fem)
        if self._use_scipy_interp:
            interp = LinearNDInterpolator(self.fem_nodes, u_fem)
            return interp(self.obs_points)
        return self.P @ u_fem

    def apply_adjoint(self, residual):
        residual = np.asarray(residual)
        if self._use_scipy_interp:
            rhs = np.zeros(self.n_fem)
            for i, pt in enumerate(self.obs_points):
                dists = np.linalg.norm(self.fem_nodes - pt, axis=1)
                rhs[np.argmin(dists)] += residual[i]
            return rhs
        return self.P.T @ residual


def create_observation_grid(n_x=6, n_y=6, margin=0.1):
    x_obs = np.linspace(margin, 1 - margin, n_x)
    y_obs = np.linspace(margin, 1 - margin, n_y)
    X_obs, Y_obs = np.meshgrid(x_obs, y_obs)
    return np.column_stack([X_obs.ravel(), Y_obs.ravel()])


# =============================================================================
# Inverse solver
# =============================================================================

class DarcyInverse:
    """
    Fast adjoint-based inverse solver for Darcy flow (piecewise-constant m on elements).
    """

    def __init__(self, nx=32, ny=32, element_degree=2,
                 f_given=None, beta=1e-4, gamma=0.0, solver_type='direct',
                 obs_points=None):
        self.nx = int(nx)
        self.ny = int(ny)
        self.element_degree = int(element_degree)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.solver_type = str(solver_type)
        self.h = 1.0 / self.nx

        if f_given is None:
            self.f_given = lambda x, y: np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
        else:
            self.f_given = f_given

        self._create_mesh()
        self._precompute_element_matrices()
        self._precompute_load_vector()
        self._setup_boundary_dofs()

        # Speedup precomputations
        self._precompute_global_assembly_indices()
        self._pack_element_tensors()

        self.obs_points = obs_points
        self.obs_operator = None
        if obs_points is not None:
            self._setup_observation_operator(obs_points)

        self.history = {'J': [], 'grad_norm': [], 'm_norm': [],
                        'data_misfit': [], 'reg_term': [], 'm_iterates': []}

    # -------------------------------------------------------------------------
    # Mesh + FE basics
    # -------------------------------------------------------------------------

    def _create_mesh(self):
        nodes_p1, elements_p1, boundary_p1 = create_triangular_mesh(self.nx, self.ny)

        if self.element_degree == 1:
            self.nodes = nodes_p1
            self.elements = elements_p1
            self.boundary_nodes = boundary_p1
        elif self.element_degree == 2:
            self.nodes, self.elements, self.boundary_nodes = create_p2_mesh(nodes_p1, elements_p1)
        else:
            raise ValueError("Only element_degree=1 or 2 supported")

        self.nodes = np.asarray(self.nodes)
        self.elements = np.asarray(self.elements, dtype=np.int32)
        self.boundary_nodes = np.asarray(self.boundary_nodes, dtype=np.int32)

        self.n_nodes = self.nodes.shape[0]
        self.n_dofs = self.n_nodes
        self.n_elements = self.elements.shape[0]

        self.element_centroids = np.zeros((self.n_elements, 2))
        for e, elem_nodes in enumerate(self.elements):
            coords = self.nodes[elem_nodes[:3]]
            self.element_centroids[e] = coords.mean(axis=0)

    def _setup_observation_operator(self, obs_points):
        self.obs_points = np.asarray(obs_points)
        self.obs_operator = ObservationOperator(
            self.nodes, self.obs_points,
            elements=self.elements, element_degree=self.element_degree
        )
        self.n_obs = self.obs_points.shape[0]
        print(f"Observation operator created: {self.n_obs} observation points")

    def _p1_basis_grad(self):
        return np.array([[-1, -1],
                         [ 1,  0],
                         [ 0,  1]], dtype=float)

    def _p2_basis_and_grad(self, xi, eta):
        lam0 = 1 - xi - eta
        lam1 = xi
        lam2 = eta

        phi = np.array([
            lam0 * (2*lam0 - 1),
            lam1 * (2*lam1 - 1),
            lam2 * (2*lam2 - 1),
            4 * lam0 * lam1,
            4 * lam1 * lam2,
            4 * lam2 * lam0
        ], dtype=float)

        dphi_dxi = np.array([
            -4*lam0 + 1,
            4*lam1 - 1,
            0,
            4*(1 - 2*xi - eta),
            4*eta,
            -4*eta
        ], dtype=float)

        dphi_deta = np.array([
            -4*lam0 + 1,
            0,
            4*lam2 - 1,
            -4*xi,
            4*xi,
            4*(1 - xi - 2*eta)
        ], dtype=float)

        grad_phi = np.column_stack([dphi_dxi, dphi_deta])
        return phi, grad_phi

    def _get_quadrature_p2(self):
        quad_points = np.array([
            [0.3333333333333333, 0.3333333333333333],
            [0.7974269853530873, 0.1012865073234563],
            [0.1012865073234563, 0.7974269853530873],
            [0.1012865073234563, 0.1012865073234563],
            [0.0597158717897698, 0.4701420641051151],
            [0.4701420641051151, 0.0597158717897698],
            [0.4701420641051151, 0.4701420641051151]
        ], dtype=float)
        quad_weights = (np.array([
            0.2250000000000000,
            0.1259391805448271,
            0.1259391805448271,
            0.1259391805448271,
            0.1323941527885062,
            0.1323941527885062,
            0.1323941527885062
        ], dtype=float) / 2.0)
        return quad_points, quad_weights

    def _compute_element_stiffness_p1(self, elem_nodes):
        coords = self.nodes[elem_nodes]
        J = np.array([coords[1] - coords[0],
                      coords[2] - coords[0]]).T
        det_J = abs(np.linalg.det(J))
        J_inv = np.linalg.inv(J)

        grad_phi = self._p1_basis_grad() @ J_inv
        A_elem = det_J / 2 * (grad_phi @ grad_phi.T)
        return A_elem, det_J

    def _compute_element_stiffness_p2(self, elem_nodes):
        coords = self.nodes[elem_nodes]
        coords_vertices = coords[:3]
        J = np.array([coords_vertices[1] - coords_vertices[0],
                      coords_vertices[2] - coords_vertices[0]]).T
        det_J = abs(np.linalg.det(J))
        J_inv = np.linalg.inv(J)

        quad_points, quad_weights = self._get_quadrature_p2()
        A_elem = np.zeros((6, 6), dtype=float)
        for (xi, eta), w in zip(quad_points, quad_weights):
            _, grad_phi_ref = self._p2_basis_and_grad(xi, eta)
            grad_phi = grad_phi_ref @ J_inv
            A_elem += det_J * w * (grad_phi @ grad_phi.T)
        return A_elem, det_J

    def _compute_element_load_p1(self, elem_nodes):
        coords = self.nodes[elem_nodes]
        J = np.array([coords[1] - coords[0],
                      coords[2] - coords[0]]).T
        det_J = abs(np.linalg.det(J))

        x_c, y_c = coords[:, 0].mean(), coords[:, 1].mean()
        f_val = self.f_given(x_c, y_c)
        F_elem = det_J / 2 * f_val * np.ones(3) / 3
        return F_elem

    def _compute_element_load_p2(self, elem_nodes):
        coords = self.nodes[elem_nodes]
        coords_vertices = coords[:3]
        J = np.array([coords_vertices[1] - coords_vertices[0],
                      coords_vertices[2] - coords_vertices[0]]).T
        det_J = abs(np.linalg.det(J))

        quad_points, quad_weights = self._get_quadrature_p2()
        F_elem = np.zeros(6, dtype=float)

        for (xi, eta), w in zip(quad_points, quad_weights):
            phi, _ = self._p2_basis_and_grad(xi, eta)
            x_phys = coords[:, 0] @ phi
            y_phys = coords[:, 1] @ phi
            f_val = self.f_given(x_phys, y_phys)
            F_elem += det_J * w * f_val * phi
        return F_elem

    def _precompute_element_matrices(self):
        self.A_elem_list = []
        for elem_nodes in self.elements:
            if self.element_degree == 1:
                A_elem, det_J = self._compute_element_stiffness_p1(elem_nodes[:3])
                self.A_elem_list.append((A_elem, det_J, elem_nodes[:3]))
            else:
                A_elem, det_J = self._compute_element_stiffness_p2(elem_nodes)
                self.A_elem_list.append((A_elem, det_J, elem_nodes))

    def _precompute_load_vector(self):
        self.F_global = np.zeros(self.n_dofs, dtype=float)
        for elem_nodes in self.elements:
            if self.element_degree == 1:
                F_elem = self._compute_element_load_p1(elem_nodes[:3])
                loc = elem_nodes[:3]
            else:
                F_elem = self._compute_element_load_p2(elem_nodes)
                loc = elem_nodes
            self.F_global[loc] += F_elem

    def _setup_boundary_dofs(self):
        self.boundary_set = set(map(int, self.boundary_nodes))
        self.interior_dofs = np.array([i for i in range(self.n_dofs) if i not in self.boundary_set],
                                      dtype=np.int32)
        self._build_element_neighbors()

    def _build_element_neighbors(self):
        from collections import defaultdict
        edge_to_elements = defaultdict(list)

        for e_idx, elem_nodes in enumerate(self.elements):
            verts = elem_nodes[:3]
            edges = [tuple(sorted((int(verts[0]), int(verts[1])))),
                     tuple(sorted((int(verts[1]), int(verts[2])))),
                     tuple(sorted((int(verts[2]), int(verts[0]))))]
            for edge in edges:
                edge_to_elements[edge].append(e_idx)

        self.element_neighbors = []
        for edge, elems in edge_to_elements.items():
            if len(elems) == 2:
                e_i, e_j = elems
                n1, n2 = edge
                edge_length = np.linalg.norm(self.nodes[n1] - self.nodes[n2])
                self.element_neighbors.append((int(e_i), int(e_j), float(edge_length)))
        self.n_internal_edges = len(self.element_neighbors)

    # -------------------------------------------------------------------------
    # SPEEDUP precomputations (assembly indices, packed tensors)
    # -------------------------------------------------------------------------

    def _precompute_global_assembly_indices(self):
        nloc = 3 if self.element_degree == 1 else 6
        nnz_per_elem = nloc * nloc

        I = np.empty(self.n_elements * nnz_per_elem, dtype=np.int32)
        J = np.empty_like(I)
        A0 = np.empty(self.n_elements * nnz_per_elem, dtype=np.float64)

        ptr = 0
        for e, (A_elem, _, elem_nodes) in enumerate(self.A_elem_list):
            elem_nodes = np.asarray(elem_nodes, dtype=np.int32)
            ii, jj = np.meshgrid(elem_nodes, elem_nodes, indexing='ij')
            I[ptr:ptr + nnz_per_elem] = ii.ravel()
            J[ptr:ptr + nnz_per_elem] = jj.ravel()
            A0[ptr:ptr + nnz_per_elem] = A_elem.ravel()
            ptr += nnz_per_elem

        self._I_all = I
        self._J_all = J
        self._A0_all = A0
        self._nnz_per_elem = nnz_per_elem

    def _pack_element_tensors(self):
        self.elem_nodes_arr = np.array([t[2] for t in self.A_elem_list], dtype=np.int32)
        self.A0_tensor = np.stack([t[0] for t in self.A_elem_list], axis=0).astype(np.float64)

    # -------------------------------------------------------------------------
    # Assembly + solves
    # -------------------------------------------------------------------------

    def assemble_stiffness(self, m):
        k = np.exp(np.asarray(m, dtype=np.float64))
        data = self._A0_all * np.repeat(k, self._nnz_per_elem)
        A = coo_matrix((data, (self._I_all, self._J_all)), shape=(self.n_dofs, self.n_dofs)).tocsr()
        A.sum_duplicates()
        return A

    def _solve_linear(self, A, b, transpose=False):
        if transpose:
            Aop = A.T
        else:
            Aop = A

        if self.solver_type == 'direct':
            return spsolve(Aop, b)

        if self.solver_type == 'amg':
            if HAS_PYAMG:
                ml = pyamg.smoothed_aggregation_solver(Aop)
                return ml.solve(b, tol=1e-10, maxiter=2000)
            return spsolve(Aop, b)

        raise ValueError(f"Unknown solver_type: {self.solver_type}")

    def solve_forward(self, m, return_matrix=False):
        A = self.assemble_stiffness(m)
        I = self.interior_dofs
        A_II = A[I][:, I]
        b_I = self.F_global[I]

        u_I = self._solve_linear(A_II, b_I, transpose=False)

        u = np.zeros(self.n_dofs, dtype=np.float64)
        u[I] = u_I
        # boundary dofs are 0 by construction

        if return_matrix:
            return u, A_II, I
        return u

    def observe(self, u_full):
        if self.obs_operator is None:
            return np.asarray(u_full).copy()
        return self.obs_operator.apply(u_full)

    def solve_adjoint(self, A_II, I, residual_obs):
        if self.obs_operator is None:
            rhs_full = np.asarray(residual_obs).copy()
        else:
            rhs_full = self.obs_operator.apply_adjoint(residual_obs)

        rhs_I = rhs_full[I]
        lam_I = self._solve_linear(A_II, rhs_I, transpose=True)

        lam = np.zeros(self.n_dofs, dtype=np.float64)
        lam[I] = lam_I
        return lam

    # -------------------------------------------------------------------------
    # Objective + gradient
    # -------------------------------------------------------------------------

    def compute_gradient(self, m, u, lam):
        m = np.asarray(m, dtype=np.float64)
        k = np.exp(m)

        nodes = self.elem_nodes_arr   # (E,nloc)
        u_loc = u[nodes]              # (E,nloc)
        lam_loc = lam[nodes]          # (E,nloc)

        sens = -k * np.einsum('ei,eij,ej->e', lam_loc, self.A0_tensor, u_loc)
        grad = sens + self.beta * m

        if self.gamma > 0:
            for e_i, e_j, edge_length in self.element_neighbors:
                diff = m[e_i] - m[e_j]
                g = self.gamma * diff / edge_length
                grad[e_i] += g
                grad[e_j] -= g

        return grad

    def objective_and_gradient(self, m, y_obs):
        u, A_II, I = self.solve_forward(m, return_matrix=True)
        u_obs = self.observe(u)
        residual_obs = u_obs - y_obs

        data_misfit = 0.5 * np.dot(residual_obs, residual_obs)
        l2_reg = 0.5 * self.beta * np.dot(m, m)

        h1_reg = 0.0
        if self.gamma > 0:
            acc = 0.0
            for e_i, e_j, edge_length in self.element_neighbors:
                diff = m[e_i] - m[e_j]
                acc += (diff * diff) / edge_length
            h1_reg = 0.5 * self.gamma * acc

        reg_term = l2_reg + h1_reg
        J = data_misfit + reg_term

        lam = self.solve_adjoint(A_II, I, residual_obs)
        grad = self.compute_gradient(m, u, lam)

        return J, grad, data_misfit, reg_term

    # -------------------------------------------------------------------------
    # Optimization driver
    # -------------------------------------------------------------------------

    def solve_inverse(self, y_obs, m0=None, method='L-BFGS-B',
                      maxiter=300, tol=1e-8, verbose=True, callback=None):
        if m0 is None:
            m0 = np.zeros(self.n_elements, dtype=np.float64)
        else:
            m0 = np.asarray(m0, dtype=np.float64)

        self.history = {'J': [], 'grad_norm': [], 'm_norm': [],
                        'data_misfit': [], 'reg_term': [], 'm_iterates': []}
        self.iteration = 0
        t_start = time.time()

        def obj_and_grad(m):
            J, g, mis, reg = self.objective_and_gradient(m, y_obs)
            self.history['J'].append(J)
            self.history['grad_norm'].append(np.linalg.norm(g))
            self.history['m_norm'].append(np.linalg.norm(m))
            self.history['data_misfit'].append(mis)
            self.history['reg_term'].append(reg)
            self.history['m_iterates'].append(np.asarray(m).copy())

            if verbose and (self.iteration % 10 == 0):
                print(f"  iter {self.iteration:4d}: J={J:.6e}, |g|={np.linalg.norm(g):.3e}, "
                      f"misfit={mis:.3e}, reg={reg:.3e}")
            self.iteration += 1
            return J, g

        def scipy_callback(m):
            if callback is not None:
                callback(m)

        if verbose:
            print("=" * 70)
            print("DARCY INVERSE (FAST)")
            print(f"  mesh: {self.nx}x{self.ny}, P{self.element_degree}")
            if self.obs_operator is None:
                print(f"  observations: full ({self.n_dofs})")
            else:
                print(f"  observations: sparse ({self.n_obs})")
            print(f"  reg: beta={self.beta:.2e}, gamma={self.gamma:.2e}")
            print(f"  solver_type: {self.solver_type}, method: {method}, maxiter: {maxiter}")
            print("=" * 70)

        result = minimize(
            obj_and_grad, m0, method=method, jac=True,
            options={'maxiter': maxiter, 'disp': verbose, 'ftol': tol},
            callback=scipy_callback
        )

        if verbose:
            print("=" * 70)
            print(f"Done in {time.time() - t_start:.2f}s")
            print(f"  success: {result.success}")
            print(f"  nit: {result.nit}")
            print(f"  final J: {result.fun:.6e}")
            print(f"  message: {result.message}")
            print("=" * 70)

        return result.x, result

    # -------------------------------------------------------------------------
    # Reference patterns + observations
    # -------------------------------------------------------------------------

    def generate_reference_m(self, pattern='rectangles', **kwargs):
        m_true = np.zeros(self.n_elements, dtype=np.float64)
        c = self.element_centroids

        if pattern == 'smooth_sine':
            amplitude = kwargs.get('amplitude', 1.0)
            kx = kwargs.get('kx', 1)
            ky = kwargs.get('ky', 1)
            m_true = amplitude * np.sin(2*np.pi*kx*c[:, 0]) * np.sin(2*np.pi*ky*c[:, 1])

        elif pattern == 'smooth_gaussian':
            gaussians = kwargs.get('gaussians', [
                {'center': (0.3, 0.3), 'amplitude': 1.0, 'width': 0.15},
                {'center': (0.7, 0.7), 'amplitude': -1.0, 'width': 0.15},
                {'center': (0.5, 0.5), 'amplitude': 0.5, 'width': 0.2},
            ])
            for g in gaussians:
                dx = c[:, 0] - g['center'][0]
                dy = c[:, 1] - g['center'][1]
                r2 = dx*dx + dy*dy
                m_true += g['amplitude'] * np.exp(-r2 / (2 * g['width']**2))

        elif pattern == 'smooth_peaks':
            amplitude = kwargs.get('amplitude', 1.0)
            x = (c[:, 0] - 0.5) * 6
            y = (c[:, 1] - 0.5) * 6
            m_true = amplitude * (3*(1-x)**2*np.exp(-x**2-(y+1)**2)
                                  - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)
                                  - (1/3)*np.exp(-(x+1)**2 - y**2)) / 10

        elif pattern == 'smooth_radial':
            amplitude = kwargs.get('amplitude', 1.0)
            center = kwargs.get('center', (0.5, 0.5))
            frequency = kwargs.get('frequency', 3)
            r = np.sqrt((c[:, 0] - center[0])**2 + (c[:, 1] - center[1])**2)
            m_true = amplitude * np.cos(2*np.pi*frequency*r)

        elif pattern == 'rectangles':
            rects = kwargs.get('rects', [
                {'x': (0.2, 0.4), 'y': (0.2, 0.4), 'm': 1.0},
                {'x': (0.6, 0.8), 'y': (0.6, 0.8), 'm': -1.0},
            ])
            for r in rects:
                mask = ((c[:, 0] >= r['x'][0]) & (c[:, 0] <= r['x'][1]) &
                        (c[:, 1] >= r['y'][0]) & (c[:, 1] <= r['y'][1]))
                m_true[mask] = r['m']

        elif pattern == 'disks':
            disks = kwargs.get('disks', [
                {'center': (0.3, 0.3), 'radius': 0.15, 'm': 1.0},
                {'center': (0.7, 0.7), 'radius': 0.15, 'm': -1.0},
                {'center': (0.3, 0.7), 'radius': 0.10, 'm': 0.5},
            ])
            for d in disks:
                dist = np.sqrt((c[:, 0] - d['center'][0])**2 + (c[:, 1] - d['center'][1])**2)
                m_true[dist <= d['radius']] = d['m']

        elif pattern == 'checkerboard':
            n_checks = int(kwargs.get('n_checks', 4))
            m_high = kwargs.get('m_high', 0.5)
            m_low = kwargs.get('m_low', -0.5)
            ix = np.floor(c[:, 0] * n_checks).astype(int)
            iy = np.floor(c[:, 1] * n_checks).astype(int)
            even = (ix + iy) % 2 == 0
            m_true[even] = m_high
            m_true[~even] = m_low

        elif pattern == 'layered':
            n_layers = int(kwargs.get('n_layers', 3))
            m_values = kwargs.get('m_values', [0.5, -0.5, 0.5])
            for i in range(n_layers):
                y0, y1 = i / n_layers, (i + 1) / n_layers
                mask = (c[:, 1] >= y0) & (c[:, 1] < y1)
                m_true[mask] = m_values[i % len(m_values)]

        elif pattern == 'random_inclusions':
            n_incl = int(kwargs.get('n_inclusions', 5))
            m_range = kwargs.get('m_range', (-1.0, 1.0))
            r_range = kwargs.get('r_range', (0.05, 0.15))
            seed = kwargs.get('seed', 42)
            rng = np.random.default_rng(seed)
            for _ in range(n_incl):
                cx = 0.1 + 0.8 * rng.random()
                cy = 0.1 + 0.8 * rng.random()
                r = r_range[0] + (r_range[1] - r_range[0]) * rng.random()
                m_val = m_range[0] + (m_range[1] - m_range[0]) * rng.random()
                dist = np.sqrt((c[:, 0] - cx)**2 + (c[:, 1] - cy)**2)
                m_true[dist <= r] = m_val

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return m_true

    def generate_observations(self, m_true, noise_level=0.0, noise_type='max', seed=None):
        u_true = self.solve_forward(m_true)
        u_obs_clean = self.observe(u_true)

        if noise_level > 0:
            rng = np.random.default_rng(seed)
            if noise_type == 'max':
                noise_mag = noise_level * np.max(np.abs(u_true))
            elif noise_type == 'std':
                noise_mag = noise_level * np.std(u_true)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_obs = u_obs_clean + noise_mag * rng.standard_normal(u_obs_clean.shape[0])
        else:
            y_obs = u_obs_clean.copy()

        return y_obs, u_true

    # -------------------------------------------------------------------------
    # Visualization helper
    # -------------------------------------------------------------------------

    def plot_elementwise_field(self, m, ax=None, cmap='RdBu_r', vmin=None, vmax=None,
                               title='Elementwise field', show_colorbar=True):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import Normalize

        m = np.asarray(m)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()

        triangles = [self.nodes[elem[:3]] for elem in self.elements]

        if vmin is None: vmin = float(np.min(m))
        if vmax is None: vmax = float(np.max(m))
        if vmin < 0 < vmax:
            a = max(abs(vmin), abs(vmax))
            vmin, vmax = -a, a

        norm = Normalize(vmin=vmin, vmax=vmax)
        pc = PolyCollection(triangles, array=m, cmap=cmap, norm=norm,
                            edgecolors='none', linewidths=0.0)
        ax.add_collection(pc)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(title)

        if show_colorbar:
            cbar = fig.colorbar(pc, ax=ax, shrink=0.8)
            return fig, ax, pc, cbar
        return fig, ax, pc

    def compute_relative_error(self, m_true, m_rec):
        return np.linalg.norm(m_rec - m_true) / (np.linalg.norm(m_true) + 1e-12)


# =============================================================================
# Convenience runner
# =============================================================================

def run_adjoint_inversion(nx=32, ny=32, element_degree=2, beta=1e-4, gamma=0.0,
                          pattern='disks', noise_level=0.0, noise_type='max',
                          obs_points=None, maxiter=200, solver_type='direct', verbose=True):
    print("=" * 70)
    print("DARCY INVERSE PROBLEM - ADJOINT METHOD (FAST)")
    print("=" * 70)
    print(f"  Mesh: {nx}×{ny}, Element: P{element_degree}")
    print(f"  Pattern: {pattern}, Noise: {noise_level*100:.1f}% ({noise_type})")
    if obs_points is not None:
        print(f"  Observations: {len(obs_points)} sparse points")
    else:
        print("  Observations: Full state")
    print(f"  Regularization: beta={beta:.2e}, gamma={gamma:.2e}")
    print(f"  Linear solver: {solver_type} (pyamg={HAS_PYAMG})")
    print("=" * 70 + "\n")

    inv = DarcyInverse(nx=nx, ny=ny, element_degree=element_degree,
                       beta=beta, gamma=gamma, solver_type=solver_type,
                       obs_points=obs_points)

    m_true = inv.generate_reference_m(pattern=pattern)
    y_obs, u_true = inv.generate_observations(m_true, noise_level=noise_level,
                                              noise_type=noise_type, seed=42)
    m_rec, result = inv.solve_inverse(y_obs, maxiter=maxiter, verbose=verbose)

    rel_error = inv.compute_relative_error(m_true, m_rec)
    print(f"\nRelative L2 error: {rel_error*100:.2f}%")
    return inv, m_true, m_rec, y_obs, result


def save_adjoint_results(inv, m_true, m_rec, y_obs, y_clean, u_clean,
                         obs_points, rel_error, t_total,
                         filepath='history/adjoint_results.npz', extra_data=None):
    """Save adjoint inversion results and history needed by plotting cells."""
    save_dict = {
        'm_true': np.asarray(m_true),
        'm_rec': np.asarray(m_rec),
        'y_obs': np.asarray(y_obs),
        'y_clean': np.asarray(y_clean),
        'u_clean': np.asarray(u_clean),
        'obs_points': np.asarray(obs_points),
        'nodes': np.asarray(inv.nodes),
        'element_centroids': np.asarray(inv.element_centroids),
        'history_J': np.asarray(inv.history.get('J', [])),
        'history_grad_norm': np.asarray(inv.history.get('grad_norm', [])),
        'history_m_norm': np.asarray(inv.history.get('m_norm', [])),
        'history_data_misfit': np.asarray(inv.history.get('data_misfit', [])),
        'history_reg_term': np.asarray(inv.history.get('reg_term', [])),
        'history_m_error': np.asarray(inv.history.get('m_error', [])),
        'beta': float(inv.beta),
        'gamma': float(inv.gamma),
        'nx': int(inv.nx),
        'ny': int(inv.ny),
        'element_degree': int(inv.element_degree),
        'n_obs': int(len(obs_points)),
        'n_elements': int(inv.n_elements),
        'n_dofs': int(inv.n_dofs),
        'rel_error': float(rel_error),
        't_total': float(t_total),
    }

    if extra_data is not None:
        save_dict.update(extra_data)

    np.savez(filepath, **save_dict)
    print(f"Adjoint results saved to {filepath}")


if __name__ == '__main__':
    obs_points = create_observation_grid(n_x=6, n_y=6, margin=0.1)

    inv, m_true, m_rec, y_obs, result = run_adjoint_inversion(
        nx=32, ny=32, element_degree=2, beta=1e-5, gamma=0.0,
        pattern='smooth_sine', noise_level=0.01, noise_type='max',
        obs_points=obs_points, maxiter=200, solver_type='direct', verbose=True
    )

    import os
    import matplotlib.pyplot as plt
    os.makedirs('figures', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    inv.plot_elementwise_field(m_true, ax=axes[0], title='True m†')
    inv.plot_elementwise_field(m_rec, ax=axes[1], title='Recovered m')
    inv.plot_elementwise_field(m_rec - m_true, ax=axes[2], title='Error')
    plt.tight_layout()
    plt.savefig('figures/adjoint_sparse_test_fast.png', dpi=150)
    plt.show()

