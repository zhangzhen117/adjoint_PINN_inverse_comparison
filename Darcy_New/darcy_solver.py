"""
Darcy Flow Forward Solver with FEM (P2 elements)
================================================

Problem:
    -∇·(k(x)∇u(x)) = f(x)  in Ω = (0,1)²
    u = 0                   on ∂Ω

where k(x) = exp(m(x)) with smooth heterogeneous m(x).

Manufactured solution for convergence testing:
    u*(x,y) = sin(πx)sin(πy)
    
Pure numpy/scipy implementation (no FEniCS).
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, cg
import time

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False
    print("Warning: pyamg not available. Install with 'pip install pyamg' for AMG preconditioner.")


def create_triangular_mesh(nx, ny):
    """
    Create a structured triangular mesh on [0,1]×[0,1].
    
    Each rectangle is divided into 2 triangles.
    
    Parameters
    ----------
    nx, ny : int
        Number of subdivisions in x and y
        
    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates
    elements : ndarray, shape (n_elements, 3)
        Element connectivity (P1 nodes)
    boundary_nodes : ndarray
        Indices of boundary nodes
    """
    # Create node grid
    x = np.linspace(0, 1, nx+1)
    y = np.linspace(0, 1, ny+1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Create elements (2 triangles per rectangle)
    elements = []
    for j in range(ny):
        for i in range(nx):
            # Node indices for rectangle corners
            n0 = j * (nx+1) + i
            n1 = n0 + 1
            n2 = n0 + (nx+1)
            n3 = n2 + 1
            
            # Two triangles
            elements.append([n0, n1, n2])  # Lower triangle
            elements.append([n1, n3, n2])  # Upper triangle
    
    elements = np.array(elements, dtype=int)
    
    # Find boundary nodes
    n_nodes = len(nodes)
    boundary_mask = (
        (nodes[:, 0] == 0) | (nodes[:, 0] == 1) |
        (nodes[:, 1] == 0) | (nodes[:, 1] == 1)
    )
    boundary_nodes = np.where(boundary_mask)[0]
    
    return nodes, elements, boundary_nodes


def create_p2_mesh(nodes_p1, elements_p1):
    """
    Add edge midpoints to P1 mesh to create P2 mesh.
    
    Parameters
    ----------
    nodes_p1 : ndarray, shape (n_p1, 2)
        P1 nodes
    elements_p1 : ndarray, shape (n_elem, 3)
        P1 element connectivity
        
    Returns
    -------
    nodes_p2 : ndarray, shape (n_p2, 2)
        All nodes (P1 + edge midpoints)
    elements_p2 : ndarray, shape (n_elem, 6)
        P2 element connectivity [v0, v1, v2, e01, e12, e20]
    boundary_nodes_p2 : ndarray
        Indices of boundary nodes (including edge midpoints)
    """
    n_p1 = len(nodes_p1)
    n_elem = len(elements_p1)
    
    # Create edge map: (i,j) -> edge_index
    edges = {}
    edge_nodes = []
    edge_index = n_p1
    
    elements_p2 = np.zeros((n_elem, 6), dtype=int)
    
    for elem_idx, elem in enumerate(elements_p1):
        v0, v1, v2 = elem
        
        # Store vertex nodes
        elements_p2[elem_idx, 0] = v0
        elements_p2[elem_idx, 1] = v1
        elements_p2[elem_idx, 2] = v2
        
        # Process edges (order matters for consistent orientation)
        for local_edge, (i, j) in enumerate([(v0, v1), (v1, v2), (v2, v0)]):
            edge = tuple(sorted([i, j]))
            
            if edge not in edges:
                # Create new edge midpoint
                midpoint = 0.5 * (nodes_p1[i] + nodes_p1[j])
                edge_nodes.append(midpoint)
                edges[edge] = edge_index
                edge_index += 1
            
            # Store edge node index
            elements_p2[elem_idx, 3 + local_edge] = edges[edge]
    
    # Combine P1 nodes and edge nodes
    nodes_p2 = np.vstack([nodes_p1, edge_nodes])
    
    # Find boundary nodes (including edge midpoints on boundary)
    boundary_mask = (
        (nodes_p2[:, 0] < 1e-10) | (nodes_p2[:, 0] > 1-1e-10) |
        (nodes_p2[:, 1] < 1e-10) | (nodes_p2[:, 1] > 1-1e-10)
    )
    boundary_nodes_p2 = np.where(boundary_mask)[0]
    
    return nodes_p2, elements_p2, boundary_nodes_p2


class DarcySolver:
    """Forward Darcy solver using FEM with P1 or P2 elements."""
    
    def __init__(self, nx=32, ny=32, alpha=0.5, element_degree=2):
        """
        Initialize Darcy solver.
        
        Parameters
        ----------
        nx, ny : int
            Number of mesh subdivisions in x and y directions
        alpha : float
            Amplitude parameter for permeability heterogeneity
            k(x,y) = exp(α*sin(2πx)*sin(2πy))
        element_degree : int
            Polynomial degree (1=P1, 2=P2)
        """
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.element_degree = element_degree
        self.h = 1.0 / nx
        
        # Create mesh
        self._create_mesh()
        
    def _create_mesh(self):
        """Create triangular mesh."""
        # Create P1 mesh
        nodes_p1, elements_p1, boundary_p1 = create_triangular_mesh(self.nx, self.ny)
        
        if self.element_degree == 1:
            self.nodes = nodes_p1
            self.elements = elements_p1
            self.boundary_nodes = boundary_p1
        elif self.element_degree == 2:
            # Create P2 mesh by adding edge midpoints
            self.nodes, self.elements, self.boundary_nodes = create_p2_mesh(nodes_p1, elements_p1)
        else:
            raise ValueError("Only element_degree=1 or 2 supported")
        
        self.n_nodes = len(self.nodes)
        self.n_dofs = self.n_nodes
        
    def k_func(self, x, y):
        """Permeability function k(x,y) = exp(α*sin(2πx)*sin(2πy))."""
        m = self.alpha * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
        return np.exp(m)
    
    def m_func(self, x, y):
        """Log-permeability m(x,y) = α*sin(2πx)*sin(2πy)."""
        return self.alpha * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    
    def u_exact_func(self, x, y):
        """Exact solution u*(x,y) = sin(πx)sin(πy)."""
        return np.sin(np.pi*x) * np.sin(np.pi*y)
    
    def grad_u_exact_func(self, x, y):
        """Gradient of exact solution."""
        du_dx = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
        du_dy = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
        return np.array([du_dx, du_dy])
    
    def f_func(self, x, y):
        """
        Forcing term from manufactured solution.
        f = -∇·(k∇u*) = -k*∇m·∇u* + 2π²k*u*
        """
        k = self.k_func(x, y)
        u_star = self.u_exact_func(x, y)
        
        # ∇m
        dm_dx = self.alpha * 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
        dm_dy = self.alpha * 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
        
        # ∇u*
        du_dx = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
        du_dy = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
        
        # f = -k*∇m·∇u* + 2π²k*u*
        f = -k * (dm_dx * du_dx + dm_dy * du_dy) + 2*np.pi**2 * k * u_star
        return f
    
    def _p1_basis_grad(self):
        """Gradient of P1 basis functions on reference triangle."""
        # Reference element: (0,0), (1,0), (0,1)
        # φ0 = 1-ξ-η, φ1 = ξ, φ2 = η
        return np.array([
            [-1, -1],  # ∇φ0
            [ 1,  0],  # ∇φ1
            [ 0,  1]   # ∇φ2
        ])
    
    def _p2_basis_and_grad(self, xi, eta):
        """
        P2 basis functions and gradients at point (ξ,η) on reference element.
        
        Reference element vertices: (0,0), (1,0), (0,1)
        P2 nodes: v0=(0,0), v1=(1,0), v2=(0,1), e01=(0.5,0), e12=(0.5,0.5), e20=(0,0.5)
        """
        lam0 = 1 - xi - eta
        lam1 = xi
        lam2 = eta
        
        # P2 basis functions (quadratic)
        phi = np.array([
            lam0 * (2*lam0 - 1),  # vertex 0
            lam1 * (2*lam1 - 1),  # vertex 1
            lam2 * (2*lam2 - 1),  # vertex 2
            4 * lam0 * lam1,       # edge 0-1
            4 * lam1 * lam2,       # edge 1-2
            4 * lam2 * lam0        # edge 2-0
        ])
        
        # Gradients w.r.t. (ξ, η)
        dphi_dxi = np.array([
            -4*lam0 + 1,
            4*lam1 - 1,
            0,
            4*(1 - 2*xi - eta),
            4*eta,
            -4*eta
        ])
        
        dphi_deta = np.array([
            -4*lam0 + 1,
            0,
            4*lam2 - 1,
            -4*xi,
            4*xi,
            4*(1 - xi - 2*eta)
        ])
        
        grad_phi = np.column_stack([dphi_dxi, dphi_deta])
        return phi, grad_phi
    
    def _assemble_element_p1(self, elem_nodes, elem_idx):
        """Assemble element stiffness matrix and load vector for P1."""
        coords = self.nodes[elem_nodes]
        
        # Jacobian of transformation
        J = np.array([
            coords[1] - coords[0],
            coords[2] - coords[0]
        ]).T
        
        det_J = np.abs(np.linalg.det(J))
        J_inv = np.linalg.inv(J)
        
        # Transform gradients to physical element
        grad_phi_ref = self._p1_basis_grad()
        grad_phi = grad_phi_ref @ J_inv
        
        # Quadrature: use centroid for k and f (exact for linear k, good for smooth k)
        x_quad = np.mean(coords[:, 0])
        y_quad = np.mean(coords[:, 1])
        k_val = self.k_func(x_quad, y_quad)
        f_val = self.f_func(x_quad, y_quad)
        
        # Element stiffness: K_ij = ∫ k ∇φi·∇φj dx
        K_elem = det_J / 2 * k_val * (grad_phi @ grad_phi.T)
        
        # Element load: F_i = ∫ f φi dx
        # For P1, φi at centroid = 1/3
        F_elem = det_J / 2 * f_val * np.ones(3) / 3
        
        return K_elem, F_elem
    
    def _assemble_element_p2(self, elem_nodes, elem_idx):
        """Assemble element stiffness matrix and load vector for P2."""
        coords = self.nodes[elem_nodes]
        
        # Get vertex coordinates for Jacobian
        coords_vertices = coords[:3]
        J = np.array([
            coords_vertices[1] - coords_vertices[0],
            coords_vertices[2] - coords_vertices[0]
        ]).T
        
        det_J = np.abs(np.linalg.det(J))
        J_inv = np.linalg.inv(J)
        
        # Higher-order Gauss quadrature for triangle (order 5, 7 points)
        # Needed for accurate integration with variable k = exp(m)
        quad_points = np.array([
            [0.3333333333333333, 0.3333333333333333],
            [0.7974269853530873, 0.1012865073234563],
            [0.1012865073234563, 0.7974269853530873],
            [0.1012865073234563, 0.1012865073234563],
            [0.0597158717897698, 0.4701420641051151],
            [0.4701420641051151, 0.0597158717897698],
            [0.4701420641051151, 0.4701420641051151]
        ])
        quad_weights = np.array([
            0.2250000000000000,
            0.1259391805448271,
            0.1259391805448271,
            0.1259391805448271,
            0.1323941527885062,
            0.1323941527885062,
            0.1323941527885062
        ]) / 2.0  # Divide by 2 for reference triangle area
        
        K_elem = np.zeros((6, 6))
        F_elem = np.zeros(6)
        
        for (xi, eta), w in zip(quad_points, quad_weights):
            # Basis functions and gradients on reference element
            phi, grad_phi_ref = self._p2_basis_and_grad(xi, eta)
            
            # Transform gradients
            grad_phi = grad_phi_ref @ J_inv
            
            # Map to physical coordinates
            x_phys = coords[:, 0] @ phi
            y_phys = coords[:, 1] @ phi
            
            # Evaluate k and f
            k_val = self.k_func(x_phys, y_phys)
            f_val = self.f_func(x_phys, y_phys)
            
            # Add contributions
            K_elem += det_J * w * k_val * (grad_phi @ grad_phi.T)
            F_elem += det_J * w * f_val * phi
        
        return K_elem, F_elem
    
    def assemble_system(self):
        """Assemble global stiffness matrix and load vector."""
        K = lil_matrix((self.n_dofs, self.n_dofs))
        F = np.zeros(self.n_dofs)
        
        n_elem = len(self.elements)
        for elem_idx, elem_nodes in enumerate(self.elements):
            if self.element_degree == 1:
                K_elem, F_elem = self._assemble_element_p1(elem_nodes, elem_idx)
            else:
                K_elem, F_elem = self._assemble_element_p2(elem_nodes, elem_idx)
            
            # Add to global system
            for i, node_i in enumerate(elem_nodes):
                F[node_i] += F_elem[i]
                for j, node_j in enumerate(elem_nodes):
                    K[node_i, node_j] += K_elem[i, j]
        
        return K.tocsr(), F
    
    def apply_bc(self, K, F):
        """Apply zero Dirichlet boundary conditions using symmetric elimination."""
        K_bc = K.tolil()  # Convert to LIL for efficient modification
        F_bc = F.copy()
        
        # Symmetric enforcement: modify RHS first, then zero rows/columns
        for node in self.boundary_nodes:
            # Get column entries for this boundary node
            col_data = K_bc.getcol(node).toarray().ravel()
            
            # Modify RHS for interior nodes coupled to this boundary node
            # (subtracting K[i,j] * u_bc where u_bc = 0, so actually no modification needed)
            # But we do need to zero the coupling
            
            # Zero the row and column
            K_bc[node, :] = 0.0
            K_bc[:, node] = 0.0
            
            # Set diagonal to 1
            K_bc[node, node] = 1.0
            
            # Set RHS to boundary value (0)
            F_bc[node] = 0.0
        
        return K_bc.tocsr(), F_bc
    
    def solve(self, solver_type='amg', verbose=False):
        """
        Solve the forward Darcy problem with manufactured solution.
        
        Parameters
        ----------
        solver_type : str
            'amg' (CG+AMG), 'direct' (spsolve), or 'cg' (CG without preconditioner)
        verbose : bool
            Print solver info
            
        Returns
        -------
        u : ndarray
            Solution DOF vector
        solve_time : float
            Time taken to solve (seconds)
        """
        t_start = time.time()
        
        # Assemble
        K, F = self.assemble_system()
        
        # Apply BC
        K_bc, F_bc = self.apply_bc(K, F)
        
        t_assembly = time.time() - t_start
        
        # Solve
        t0 = time.time()
        if solver_type == 'direct':
            u = spsolve(K_bc, F_bc)
        elif solver_type == 'cg':
            u, info = cg(K_bc, F_bc, tol=1e-12, maxiter=10000)
            if info != 0:
                print(f"Warning: CG did not converge, info={info}")
        elif solver_type == 'amg':
            if HAS_PYAMG:
                ml = pyamg.smoothed_aggregation_solver(K_bc)
                u = ml.solve(F_bc, tol=1e-12, maxiter=10000)
            else:
                print("Warning: pyamg not available, falling back to direct solver")
                u = spsolve(K_bc, F_bc)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        solve_time = time.time() - t0
        
        if verbose:
            print(f"  Mesh: {self.nx}×{self.ny}, h={self.h:.4e}")
            print(f"  DOFs: {self.n_dofs}")
            print(f"  Assembly time: {t_assembly:.4f}s")
            print(f"  Solve time: {solve_time:.4f}s")
        
        return u, solve_time
    
    def compute_errors(self, u_h):
        """
        Compute L2 and H1 seminorm errors against manufactured solution.
        
        Parameters
        ----------
        u_h : ndarray
            Computed solution DOF vector
            
        Returns
        -------
        error_L2 : float
            L2 norm error ||u* - u_h||_{L2}
        error_H1 : float
            H1 seminorm error ||∇(u* - u_h)||_{L2}
        """
        error_L2_sq = 0.0
        error_H1_sq = 0.0
        
        for elem_idx, elem_nodes in enumerate(self.elements):
            coords = self.nodes[elem_nodes]
            u_elem = u_h[elem_nodes]
            
            if self.element_degree == 1:
                # P1: use centroid quadrature
                coords_vertices = coords
                J = np.array([
                    coords_vertices[1] - coords_vertices[0],
                    coords_vertices[2] - coords_vertices[0]
                ]).T
                det_J = np.abs(np.linalg.det(J))
                
                # Centroid
                x_c = np.mean(coords[:, 0])
                y_c = np.mean(coords[:, 1])
                
                # Exact values
                u_exact = self.u_exact_func(x_c, y_c)
                u_h_val = np.mean(u_elem)
                
                # L2 error
                error_L2_sq += det_J / 2 * (u_exact - u_h_val)**2
                
                # H1 error: need gradient
                J_inv = np.linalg.inv(J)
                grad_phi_ref = self._p1_basis_grad()
                grad_phi = grad_phi_ref @ J_inv
                
                grad_u_h = grad_phi.T @ u_elem
                grad_u_exact = self.grad_u_exact_func(x_c, y_c)
                grad_error = grad_u_exact - grad_u_h
                
                error_H1_sq += det_J / 2 * np.dot(grad_error, grad_error)
                
            else:  # P2
                coords_vertices = coords[:3]
                J = np.array([
                    coords_vertices[1] - coords_vertices[0],
                    coords_vertices[2] - coords_vertices[0]
                ]).T
                det_J = np.abs(np.linalg.det(J))
                J_inv = np.linalg.inv(J)
                
                # Quadrature (same higher-order rule as assembly)
                quad_points = np.array([
                    [0.3333333333333333, 0.3333333333333333],
                    [0.7974269853530873, 0.1012865073234563],
                    [0.1012865073234563, 0.7974269853530873],
                    [0.1012865073234563, 0.1012865073234563],
                    [0.0597158717897698, 0.4701420641051151],
                    [0.4701420641051151, 0.0597158717897698],
                    [0.4701420641051151, 0.4701420641051151]
                ])
                quad_weights = np.array([
                    0.2250000000000000,
                    0.1259391805448271,
                    0.1259391805448271,
                    0.1259391805448271,
                    0.1323941527885062,
                    0.1323941527885062,
                    0.1323941527885062
                ]) / 2.0
                
                for (xi, eta), w in zip(quad_points, quad_weights):
                    phi, grad_phi_ref = self._p2_basis_and_grad(xi, eta)
                    grad_phi = grad_phi_ref @ J_inv
                    
                    # Physical coordinates
                    x_phys = coords[:, 0] @ phi
                    y_phys = coords[:, 1] @ phi
                    
                    # Computed solution value and gradient
                    u_h_val = u_elem @ phi
                    grad_u_h = grad_phi.T @ u_elem
                    
                    # Exact solution value and gradient
                    u_exact = self.u_exact_func(x_phys, y_phys)
                    grad_u_exact = self.grad_u_exact_func(x_phys, y_phys)
                    
                    # Errors
                    error_L2_sq += det_J * w * (u_exact - u_h_val)**2
                    grad_error = grad_u_exact - grad_u_h
                    error_H1_sq += det_J * w * np.dot(grad_error, grad_error)
        
        return np.sqrt(error_L2_sq), np.sqrt(error_H1_sq)
    
    def sample_at_points(self, u, points):
        """
        Sample solution at given points (for inverse problem data).
        
        Uses linear/quadratic interpolation within elements.
        
        Parameters
        ----------
        u : ndarray
            Solution DOF vector
        points : array-like, shape (n_points, 2)
            Coordinates of sampling points
            
        Returns
        -------
        values : ndarray, shape (n_points,)
            Solution values at points
        """
        points = np.asarray(points)
        values = np.zeros(len(points))
        
        for i, pt in enumerate(points):
            # Find element containing point (brute force, could use spatial search)
            found = False
            for elem_idx, elem_nodes in enumerate(self.elements):
                coords = self.nodes[elem_nodes[:3]]  # Use only vertices for containment test
                
                # Barycentric coordinates
                v0 = coords[1] - coords[0]
                v1 = coords[2] - coords[0]
                v2 = pt - coords[0]
                
                d00 = np.dot(v0, v0)
                d01 = np.dot(v0, v1)
                d11 = np.dot(v1, v1)
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)
                
                denom = d00 * d11 - d01 * d01
                if abs(denom) < 1e-10:
                    continue
                    
                xi = (d11 * d20 - d01 * d21) / denom
                eta = (d00 * d21 - d01 * d20) / denom
                
                if xi >= -1e-10 and eta >= -1e-10 and xi + eta <= 1 + 1e-10:
                    # Point is in this element
                    found = True
                    u_elem = u[elem_nodes]
                    
                    if self.element_degree == 1:
                        # Linear interpolation
                        lam0 = 1 - xi - eta
                        lam1 = xi
                        lam2 = eta
                        values[i] = lam0 * u_elem[0] + lam1 * u_elem[1] + lam2 * u_elem[2]
                    else:
                        # P2 interpolation
                        phi, _ = self._p2_basis_and_grad(xi, eta)
                        values[i] = u_elem @ phi
                    break
            
            if not found:
                print(f"Warning: point {pt} not found in any element")
                values[i] = np.nan
        
        return values


def convergence_test(mesh_sizes=[8, 16, 32, 64], alpha=0.5, element_degree=2, 
                     solver_type='amg', verbose=True):
    """
    Run spatial convergence test with manufactured solution.
    
    Parameters
    ------Solve (uses manufactured solution internally)
        u_h, solve_time = solver.solve(solver_type=solver_type, verbose=verbose)
        
        # Compute errors
        error_L2, error_H1 = solver.compute_errors(u_h
        Linear solver type
    verbose : bool
        Print detailed info
        
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'h': mesh sizes
        - 'dofs': degrees of freedom
        - 'error_L2': L2 errors
        - 'error_H1': H1 seminorm errors
        - 'rate_L2': observed L2 convergence rates
        - 'rate_H1': observed H1 convergence rates
        - 'solve_time': solve times
    """
    results = {
        'h': [],
        'dofs': [],
        'error_L2': [],
        'error_H1': [],
        'solve_time': []
    }
    
    if verbose:
        print("="*70)
        print(f"Darcy Flow Convergence Test")
        print(f"Element: P{element_degree} (CG{element_degree})")
        print(f"Solver: {solver_type.upper()}")
        print(f"Permeability: k = exp({alpha}*sin(2πx)*sin(2πy))")
        print(f"  k_min = {np.exp(-alpha):.4f}, k_max = {np.exp(alpha):.4f}")
        print("="*70)
    
    for nx in mesh_sizes:
        if verbose:
            print(f"\nMesh: {nx}×{nx}")
            print("-"*70)
        
        # Create solver
        solver = DarcySolver(nx=nx, ny=nx, alpha=alpha, element_degree=element_degree)
        
        # Solve (uses manufactured solution internally)
        u_h, solve_time = solver.solve(solver_type=solver_type, verbose=verbose)
        
        # Compute errors
        error_L2, error_H1 = solver.compute_errors(u_h)
        
        # Store results
        results['h'].append(solver.h)
        results['dofs'].append(solver.n_dofs)
        results['error_L2'].append(error_L2)
        results['error_H1'].append(error_H1)
        results['solve_time'].append(solve_time)
        
        if verbose:
            print(f"  L2 error:  {error_L2:.6e}")
            print(f"  H1 error:  {error_H1:.6e}")
    
    # Compute convergence rates
    h_array = np.array(results['h'])
    error_L2_array = np.array(results['error_L2'])
    error_H1_array = np.array(results['error_H1'])
    
    rate_L2 = np.log(error_L2_array[:-1] / error_L2_array[1:]) / np.log(2.0)
    rate_H1 = np.log(error_H1_array[:-1] / error_H1_array[1:]) / np.log(2.0)
    
    results['rate_L2'] = rate_L2
    results['rate_H1'] = rate_H1
    
    if verbose:
        print("\n" + "="*70)
        print("CONVERGENCE SUMMARY")
        print("="*70)
        print(f"{'h':<12} {'DOFs':<8} {'L2 error':<14} {'L2 rate':<10} {'H1 error':<14} {'H1 rate':<10}")
        print("-"*70)
        for i in range(len(mesh_sizes)):
            h_str = f"{results['h'][i]:.4e}"
            dof_str = f"{results['dofs'][i]}"
            l2_str = f"{results['error_L2'][i]:.6e}"
            h1_str = f"{results['error_H1'][i]:.6e}"
            
            if i == 0:
                rate_l2_str = "---"
                rate_h1_str = "---"
            else:
                rate_l2_str = f"{rate_L2[i-1]:.3f}"
                rate_h1_str = f"{rate_H1[i-1]:.3f}"
            
            print(f"{h_str:<12} {dof_str:<8} {l2_str:<14} {rate_l2_str:<10} {h1_str:<14} {rate_h1_str:<10}")
        
        print("-"*70)
        if len(rate_L2) > 0:
            print(f"Average L2 rate: {np.mean(rate_L2):.3f} (expected: 3 for P2)")
            print(f"Average H1 rate: {np.mean(rate_H1):.3f} (expected: 2 for P2)")
        print("="*70)
    
    return results


def generate_sensor_points(n_sensors=25, margin=0.1):
    """
    Generate sensor locations for inverse problem (pointwise pressure data).
    
    Parameters
    ----------
    n_sensors : int
        Number of sensors (will be arranged in a grid)
    margin : float
        Margin from boundaries (to avoid boundary layer artifacts)
        
    Returns
    -------
    points : ndarray, shape (n_sensors, 2)
        Sensor coordinates
    """
    # Create roughly uniform grid with margin
    n_per_dim = int(np.sqrt(n_sensors))
    x = np.linspace(margin, 1-margin, n_per_dim)
    y = np.linspace(margin, 1-margin, n_per_dim)
    
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    return points[:n_sensors]


if __name__ == '__main__':
    # Run convergence test
    print("\n" + "/"*70)
    print("  Darcy Flow FEM Solver - Convergence Test")
    print("/"*70 + "\n")
    
    # Test with different element orders
    for degree in [1, 2]:
        results = convergence_test(
            mesh_sizes=[8, 16, 32, 64],
            alpha=0.5,
            element_degree=degree,
            solver_type='amg',
            verbose=True
        )
        print("\n")
    
    # Example: solve on fine mesh and sample at points
    print("\n" + "/"*70)
    print("  Example: Solve and sample at sensor points")
    print("/"*70 + "\n")
    
    solver = DarcySolver(nx=64, ny=64, alpha=0.5, element_degree=2)
    u_exact, f_expr = solver.manufactured_solution()
    u_h, _ = solver.solve(f_expr, solver_type='amg', verbose=True)
    
    # h, _ = solver.solve(solver_type='amg', verbose=True)
    
    # Generate sensor points
    sensor_points = generate_sensor_points(n_sensors=25, margin=0.1)
    u_obs = solver.sample_at_points(u_h, sensor_points)
    
    # Also get exact values at sensors
    u_exact_sensors = np.array([solver.u_exact_func(p[0], p[1]) for p in sensor_points])
    
    print(f"\nSensor points shape: {sensor_points.shape}")
    print(f"Observed values shape: {u_obs.shape}")
    print(f"Sample observed values: {u_obs[:5]}")
    print(f"Sample exact values:    {u_exact_sensors[:5]}")
    print(f"Max pointwise error:    {np.max(np.abs(u_obs - u_exact_sensors)):.6e}")