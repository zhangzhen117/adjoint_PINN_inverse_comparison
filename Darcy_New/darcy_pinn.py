"""
PINN-based inverse solver for Darcy flow with sparse observations.

Jointly learns u(x,y) and m(x,y) from sparse observations using physics-informed loss.

PDE: -∇·(k∇u) = f,  k = exp(m),  u|∂Ω = 0

Loss = L_data + L_pde + L_bc + (β/2) * R_L2(m) + (γ/2) * R_H1(m)

where:
- L_data  = (1/N_data) Σ |u_θ(x_i,y_i) - y_i|²  (at sparse observation points)
- L_pde   = (1/N_pde)  Σ |r(x,y)|²              (PDE residual)
- L_bc    = (1/N_bc)   Σ |u_θ(x,y)|²            (soft Dirichlet BC)
- R_L2(m) = (1/N_reg)  Σ |m_φ(x,y)|²            (Tikhonov / L2 on m)
- R_H1(m) = (1/N_reg)  Σ |∇m_φ(x,y)|²           (Sobolev / H1 seminorm on m)

Training: Adam → SciPy BFGS with periodic resampling.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from numpy.linalg import cholesky, LinAlgError
import time
from typing import Callable, Optional, Tuple, Dict, List

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


class MLP(nn.Module):
    """Simple fully-connected network with tanh activation."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 activation: nn.Module = nn.Tanh):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class MultiOutputMLP(nn.Module):
    """Shared-trunk network with two output heads for u and m."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 head_dims_u: List[int] = [], head_dims_m: List[int] = [],
                 activation: nn.Module = nn.Tanh):
        super().__init__()
        # Shared trunk
        trunk_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, h))
            trunk_layers.append(activation())
            prev_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = prev_dim  # last hidden dim
        
        # Head for u
        u_layers = []
        prev = trunk_out
        for h in head_dims_u:
            u_layers.append(nn.Linear(prev, h))
            u_layers.append(activation())
            prev = h
        u_layers.append(nn.Linear(prev, 1))
        self.head_u = nn.Sequential(*u_layers)
        
        # Head for m
        m_layers = []
        prev = trunk_out
        for h in head_dims_m:
            m_layers.append(nn.Linear(prev, h))
            m_layers.append(activation())
            prev = h
        m_layers.append(nn.Linear(prev, 1))
        self.head_m = nn.Sequential(*m_layers)
        
        # Xavier initialization
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)
    
    def forward(self, x):
        """Returns (u, m) predictions."""
        h = self.trunk(x)
        return self.head_u(h), self.head_m(h)
    
    def forward_u(self, x):
        h = self.trunk(x)
        return self.head_u(h)
    
    def forward_m(self, x):
        h = self.trunk(x)
        return self.head_m(h)


class DarcyPINN:
    """
    PINN solver for Darcy inverse problem with sparse observations.
    
    Learns both u(x,y) and m(x,y) from sparse observations.
    """
    
    def __init__(
        self,
        f_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data_points: np.ndarray,  # (N_data, 2) - observation point coordinates
        data_values: np.ndarray,  # (N_data,) - observed u values
        beta: float = 0.0,
        gamma: float = 0.0,
        hidden_dims: List[int] = [64, 64, 64],
        head_dims_u: List[int] = [],
        head_dims_m: List[int] = [],
        device: str = 'cpu',
        m_true_func: Callable = None  # For error tracking during training
    ):
        """
        Args:
            f_func: Source term f(x, y) -> scalar (works with torch tensors)
            data_points: Observation locations (can be sparse)
            data_values: Observed pressure values y
            beta: Tikhonov (L2) regularization parameter, penalizes mean(m²)
            gamma: Sobolev (H1) regularization parameter, penalizes mean(|∇m|²)
            hidden_dims: Hidden layer sizes for the shared trunk
            head_dims_u: Hidden layer sizes for the u output head (empty = linear)
            head_dims_m: Hidden layer sizes for the m output head (empty = linear)
            device: 'cpu' or 'cuda'
            m_true_func: Optional function to compute true m for error tracking
        """
        self.device = torch.device(device)
        self.beta = beta
        self.gamma = gamma
        self.f_func = f_func
        self.m_true_func = m_true_func
        
        # Store observation data
        self.x_data = torch.tensor(data_points[:, 0:1], dtype=torch.float64, 
                                   device=self.device)
        self.y_data = torch.tensor(data_points[:, 1:2], dtype=torch.float64,
                                   device=self.device)
        self.u_obs = torch.tensor(data_values.reshape(-1, 1), dtype=torch.float64,
                                  device=self.device)
        self.n_data = len(data_values)
        
        # Create single network with shared trunk and two output heads
        self.net = MultiOutputMLP(
            input_dim=2, hidden_dims=hidden_dims,
            head_dims_u=head_dims_u, head_dims_m=head_dims_m
        ).to(self.device)
        
        # Collocation points (will be sampled)
        self.x_pde = None
        self.y_pde = None
        self.x_bc = None
        self.y_bc = None
        self.x_reg = None
        self.y_reg = None
        
        # Training history
        self.history = {
            'loss': [],
            'loss_data': [],
            'loss_pde': [],
            'loss_bc': [],
            'loss_reg': [],
            'grad_norm': [],
            'iteration': [],
            'm_error': []  # Track m error during training
        }
        self.iter_count = 0
    
    def sample_collocation_points(
        self,
        n_pde: int = 2000,
        n_bc: int = 200,
        n_reg: int = 1000,
        seed: Optional[int] = None
    ):
        """Sample new collocation points for PDE, BC, and regularization."""
        if seed is not None:
            np.random.seed(seed)
        
        # Interior points for PDE residual (Latin hypercube or random)
        x_pde = np.random.rand(n_pde)
        y_pde = np.random.rand(n_pde)
        self.x_pde = torch.tensor(x_pde.reshape(-1, 1), dtype=torch.float64,
                                  device=self.device, requires_grad=True)
        self.y_pde = torch.tensor(y_pde.reshape(-1, 1), dtype=torch.float64,
                                  device=self.device, requires_grad=True)
        
        # Boundary points (all four edges)
        n_per_edge = n_bc // 4
        bc_points = []
        # Bottom: y = 0
        bc_points.append(np.column_stack([np.random.rand(n_per_edge), 
                                          np.zeros(n_per_edge)]))
        # Top: y = 1
        bc_points.append(np.column_stack([np.random.rand(n_per_edge), 
                                          np.ones(n_per_edge)]))
        # Left: x = 0
        bc_points.append(np.column_stack([np.zeros(n_per_edge), 
                                          np.random.rand(n_per_edge)]))
        # Right: x = 1
        bc_points.append(np.column_stack([np.ones(n_per_edge), 
                                          np.random.rand(n_per_edge)]))
        bc_points = np.vstack(bc_points)
        
        self.x_bc = torch.tensor(bc_points[:, 0:1], dtype=torch.float64,
                                 device=self.device)
        self.y_bc = torch.tensor(bc_points[:, 1:2], dtype=torch.float64,
                                 device=self.device)
        
        # Points for regularization (sample m and ∇m at these locations).
        # requires_grad=True so the H1 (Sobolev) term can autograd ∇m.
        x_reg = np.random.rand(n_reg)
        y_reg = np.random.rand(n_reg)
        self.x_reg = torch.tensor(x_reg.reshape(-1, 1), dtype=torch.float64,
                                  device=self.device, requires_grad=True)
        self.y_reg = torch.tensor(y_reg.reshape(-1, 1), dtype=torch.float64,
                                  device=self.device, requires_grad=True)
    
    def forward_u(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate u output from the shared network."""
        xy = torch.cat([x, y], dim=1)
        return self.net.forward_u(xy)
    
    def forward_m(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate m output from the shared network."""
        xy = torch.cat([x, y], dim=1)
        return self.net.forward_m(xy)
    
    def compute_pde_residual(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual: r = -∇·(k∇u) - f, where k = exp(m).
        
        Expanding: r = -k(∂²u/∂x² + ∂²u/∂y²) - (∂k/∂x)(∂u/∂x) - (∂k/∂y)(∂u/∂y) - f
                     = -k(u_xx + u_yy) - k_x*u_x - k_y*u_y - f
        
        Since k = exp(m): k_x = k*m_x, k_y = k*m_y
        So: r = -k*(u_xx + u_yy + m_x*u_x + m_y*u_y) - f
        """
        # Ensure requires_grad
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        
        # Forward pass
        u = self.forward_u(x, y)
        m = self.forward_m(x, y)
        k = torch.exp(m)
        
        # First derivatives of u
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        
        # Second derivatives of u
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                   create_graph=True)[0]
        
        # First derivatives of m
        m_x = torch.autograd.grad(m, x, grad_outputs=torch.ones_like(m),
                                  create_graph=True)[0]
        m_y = torch.autograd.grad(m, y, grad_outputs=torch.ones_like(m),
                                  create_graph=True)[0]
        
        # Source term
        f = self.f_func(x, y)
        
        # PDE residual: -∇·(k∇u) - f = -k*(u_xx + u_yy + m_x*u_x + m_y*u_y) - f
        residual = -k * (u_xx + u_yy + m_x * u_x + m_y * u_y) - f
        
        return residual
    
    def compute_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and individual components."""
        # Data loss: MSE at observation points
        u_pred = self.forward_u(self.x_data, self.y_data)
        loss_data = torch.mean((u_pred - self.u_obs) ** 2)
        
        # PDE residual loss
        residual = self.compute_pde_residual(self.x_pde, self.y_pde)
        loss_pde = torch.mean(residual ** 2)
        
        # Boundary loss (soft Dirichlet: u = 0 on ∂Ω)
        u_bc = self.forward_u(self.x_bc, self.y_bc)
        loss_bc = torch.mean(u_bc ** 2)
        
        # Regularization on m (matches adjoint objective, continuous functionals):
        #   L2 (Tikhonov):  β/2 * mean(m²)        ≈ β/2 ∫ m² dx
        #   H1 (Sobolev):   γ/2 * mean(|∇m|²)     ≈ γ/2 ∫ |∇m|² dx
        x_reg = self.x_reg.requires_grad_(True)
        y_reg = self.y_reg.requires_grad_(True)
        m_reg = self.forward_m(x_reg, y_reg)
        loss_reg = 0.5 * self.beta * torch.mean(m_reg ** 2)
        if self.gamma != 0.0:
            m_rx = torch.autograd.grad(m_reg, x_reg, grad_outputs=torch.ones_like(m_reg),
                                       create_graph=True)[0]
            m_ry = torch.autograd.grad(m_reg, y_reg, grad_outputs=torch.ones_like(m_reg),
                                       create_graph=True)[0]
            loss_reg = loss_reg + 0.5 * self.gamma * torch.mean(m_rx ** 2 + m_ry ** 2)
        
        # Total loss (all weights = 1)
        total_loss = loss_data + loss_pde + loss_bc + loss_reg
        
        components = {
            'loss': total_loss.item(),
            'loss_data': loss_data.item(),
            'loss_pde': loss_pde.item(),
            'loss_bc': loss_bc.item(),
            'loss_reg': loss_reg.item()
        }
        
        return total_loss, components
    
    def compute_m_error(self):
        """Compute relative L2 error of m on a grid."""
        if self.m_true_func is None:
            return None
        
        n_eval = 50
        x_eval = np.linspace(0, 1, n_eval)
        y_eval = np.linspace(0, 1, n_eval)
        X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
        X_flat, Y_flat = X_eval.flatten(), Y_eval.flatten()
        
        # True m on grid
        m_true_eval = self.m_true_func(X_eval, Y_eval).flatten()
        
        # PINN m on grid
        m_pinn_eval = self.predict_m(X_flat, Y_flat)
        
        # Compute error
        m_error = np.linalg.norm(m_pinn_eval - m_true_eval) / (np.linalg.norm(m_true_eval) + 1e-10)
        return m_error
    
    def train_adam(
        self,
        n_iterations: int = 5000,
        lr: float = 1e-3,
        resample_every: int = 200,
        n_pde: int = 2000,
        n_bc: int = 200,
        n_reg: int = 1000,
        verbose: bool = True,
        print_every: int = 500,
        callback: callable = None
    ):
        """
        Train using Adam optimizer with periodic resampling.
        
        Args:
            callback: Optional callback function with signature:
                      callback(pinn, iteration, loss_dict)
        """
        # All parameters from the single shared network
        params = list(self.net.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)
        
        # Initial sampling
        self.sample_collocation_points(n_pde, n_bc, n_reg)
        
        if verbose:
            print("=" * 70)
            print("PINN Training - Adam Phase")
            print("=" * 70)
            print(f"  Iterations: {n_iterations}")
            print(f"  Learning rate: {lr}")
            print(f"  Resample every: {resample_every} iterations")
            print(f"  Collocation points: {n_pde} PDE, {n_bc} BC, {n_reg} reg")
            print(f"  Observation points: {self.n_data}")
            print("-" * 70)
        
        t_start = time.time()
        
        for i in range(n_iterations):
            # Resample collocation points
            if i > 0 and i % resample_every == 0:
                self.sample_collocation_points(n_pde, n_bc, n_reg)
            
            optimizer.zero_grad()
            loss, components = self.compute_loss()
            loss.backward()
            
            # Compute gradient norm
            grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = np.sqrt(grad_norm)
            
            optimizer.step()
            
            # Compute m error
            m_error = self.compute_m_error()
            
            # Record history
            self.history['loss'].append(components['loss'])
            self.history['loss_data'].append(components['loss_data'])
            self.history['loss_pde'].append(components['loss_pde'])
            self.history['loss_bc'].append(components['loss_bc'])
            self.history['loss_reg'].append(components['loss_reg'])
            self.history['grad_norm'].append(grad_norm)
            self.history['iteration'].append(self.iter_count)
            self.history['m_error'].append(m_error if m_error is not None else np.nan)
            self.iter_count += 1
            
            if verbose and (i + 1) % print_every == 0:
                m_error_str = f", m_err={m_error*100:.2f}%" if m_error is not None else ""
                print(f"  Iter {i+1:5d}: loss={components['loss']:.4e}, "
                      f"data={components['loss_data']:.4e}, "
                      f"pde={components['loss_pde']:.4e}, "
                      f"bc={components['loss_bc']:.4e}, "
                      f"|∇|={grad_norm:.4e}{m_error_str}")
            
            # Invoke callback if provided
            if callback is not None:
                callback(self, i + 1, components)
        
        t_elapsed = time.time() - t_start
        if verbose:
            print("-" * 70)
            print(f"  Adam training completed in {t_elapsed:.2f}s")
    
    def train_bfgs(
        self,
        maxiter: int = 50,
        n_epochs: int = 10,
        n_restarts: int = 10,
        resample_every_epoch: bool = True,
        n_pde: int = 2000,
        n_bc: int = 200,
        n_reg: int = 1000,
        method_bfgs: str = 'SSBroyden2',
        gtol: float = 0.0,
        ftol: float = 1e-12,
        target_loss: float = 1e-6,
        adam_recovery_steps: int = 20,
        adam_recovery_lr: float = 1e-4,
        verbose: bool = True,
        disp: bool = False,
        print_m_error_every: int = 50  # Print m error every N iterations
    ):
        """
        Fine-tune using SciPy BFGS (SSBroyden2) with epochs and restarts.
        
        This matches the training loop from AllenCahn/PINN_control.py:
        - Outer loop: epochs with resampling
        - Inner loop: BFGS restarts with Hessian carry-over and Adam recovery
        
        Args:
            maxiter: Max iterations per BFGS call
            n_epochs: Number of outer epochs (with resampling)
            n_restarts: Max BFGS restarts per epoch (with Adam recovery)
            resample_every_epoch: Whether to resample collocation points each epoch
            n_pde: Number of interior collocation points
            n_bc: Number of boundary collocation points
            n_reg: Number of regularization points
            method_bfgs: BFGS variant ('SSBroyden2' for your patched scipy)
            gtol: Gradient tolerance
            ftol: Function tolerance
            target_loss: Stop if loss falls below this
            adam_recovery_steps: Adam steps if BFGS fails
            adam_recovery_lr: Learning rate for Adam recovery
            verbose: Print progress
            disp: scipy.optimize display flag
            print_m_error_every: Print m error every N iterations
        """
        if verbose:
            print("=" * 70)
            print("PINN Training - BFGS Phase (SSBroyden2)")
            print("=" * 70)
            print(f"  Epochs: {n_epochs}, Max restarts/epoch: {n_restarts}")
            print(f"  Max iter/restart: {maxiter}, method_bfgs: {method_bfgs}")
            print(f"  gtol: {gtol:.2e}, ftol: {ftol:.2e}, target_loss: {target_loss:.2e}")
            print(f"  Resample each epoch: {resample_every_epoch}")
            print(f"  Print m error every: {print_m_error_every} iterations")
            print("-" * 70)
        
        t_start = time.time()
        
        # Initial collocation sampling
        self.sample_collocation_points(n_pde, n_bc, n_reg)
        
        # Get initial parameters
        params_flat = self._get_params_flat()
        n_params = len(params_flat)
        
        # Initialize Hessian inverse approximation
        H0 = np.eye(n_params)
        
        # Track last evaluation for callback
        last_eval = {}
        bfgs_iter_count = [0]  # Use list to allow mutation in nested function
        
        def fun_and_jac(params_vec):
            """Combined objective and gradient for BFGS."""
            self._set_params_flat(params_vec)
            
            # Zero gradients
            self.net.zero_grad()
            
            loss, components = self.compute_loss()
            loss.backward()
            
            grad = self._get_grad_flat()
            grad_norm = np.linalg.norm(grad)
            
            # Compute m error
            m_error = self.compute_m_error()
            
            # Record history
            self.history['loss'].append(components['loss'])
            self.history['loss_data'].append(components['loss_data'])
            self.history['loss_pde'].append(components['loss_pde'])
            self.history['loss_bc'].append(components['loss_bc'])
            self.history['loss_reg'].append(components['loss_reg'])
            self.history['grad_norm'].append(grad_norm)
            self.history['iteration'].append(self.iter_count)
            self.history['m_error'].append(m_error if m_error is not None else np.nan)
            self.iter_count += 1
            bfgs_iter_count[0] += 1
            
            # Update last_eval for callback
            last_eval.update({
                'loss': components['loss'],
                'grad_norm': grad_norm,
                'm_error': m_error,
                **components
            })
            
            # Print m error every N iterations
            if verbose and bfgs_iter_count[0] % print_m_error_every == 0:
                m_error_str = f", m_err={m_error*100:.4f}%" if m_error is not None else ""
                print(f"  [BFGS iter {bfgs_iter_count[0]:4d}] loss={components['loss']:.4e}, "
                      f"data={components['loss_data']:.4e}, pde={components['loss_pde']:.4e}, "
                      f"bc={components['loss_bc']:.4e}, reg={components['loss_reg']:.4e}, "
                      f"|∇|={grad_norm:.4e}{m_error_str}")
            
            if disp:
                print(f"[BFGS eval] loss={components['loss']:.4e}, |∇|={grad_norm:.4e}")
            
            return components['loss'], grad
        
        def callback(_xk):
            """Callback after each BFGS iteration."""
            pass  # Printing is now done in fun_and_jac
        
        # Get current parameters
        theta0 = self._get_params_flat()
        
        # Outer loop: epochs with resampling
        for epoch in range(n_epochs):
            if verbose:
                print(f"\n  Epoch {epoch+1}/{n_epochs}")
            
            # Resample collocation points at the start of each epoch
            if resample_every_epoch or epoch == 0:
                self.sample_collocation_points(n_pde, n_bc, n_reg)
            
            # Inner loop: BFGS restarts with Hessian carry-over (no resampling)
            for restart in range(n_restarts):
                # Run BFGS with SSBroyden2 options
                result = minimize(
                    fun=fun_and_jac,
                    x0=theta0,
                    jac=True,
                    method='BFGS',
                    tol=ftol,
                    callback=callback,
                    options=dict(
                        maxiter=maxiter,
                        gtol=gtol,
                        disp=disp,
                        method_bfgs=method_bfgs,
                        hess_inv0=H0,
                        initial_scale=False
                    )
                )
                
                theta0 = result.x.copy()
                self._set_params_flat(theta0)
                
                # Try to carry over Hessian inverse if available and valid
                H0_new = getattr(result, 'hess_inv', None)
                if H0_new is not None:
                    try:
                        # Symmetrize and check positive definiteness
                        H0_new = 0.5 * (H0_new + H0_new.T)
                        cholesky(H0_new)
                        H0 = H0_new
                    except (LinAlgError, Exception):
                        # Reset to identity if not positive definite
                        H0 = np.eye(n_params)
                else:
                    H0 = np.eye(n_params)
                
                # Get current m error for printout
                m_error = self.compute_m_error()
                m_error_str = f", m_err={m_error*100:.2f}%" if m_error is not None else ""
                
                if verbose:
                    print(f"    Restart {restart+1}: loss={result.fun:.4e}, "
                          f"data={last_eval.get('loss_data', 0):.4e}, "
                          f"pde={last_eval.get('loss_pde', 0):.4e}, "
                          f"bc={last_eval.get('loss_bc', 0):.4e}, "
                          f"reg={last_eval.get('loss_reg', 0):.4e}, "
                          f"nit={result.nit}{m_error_str}")
                
                # Check if target loss reached
                if result.fun < target_loss:
                    if verbose:
                        print(f"  Target loss {target_loss:.2e} reached!")
                    break
                else:
                    if verbose:
                        print(f"    BFGS stalled, running {adam_recovery_steps} Adam recovery steps...")
                    self._adam_recovery(adam_recovery_steps, adam_recovery_lr)
                    theta0 = self._get_params_flat()
                    H0 = np.eye(n_params)  # Reset Hessian after Adam
            
        
        t_elapsed = time.time() - t_start
        
        # Final m error
        final_m_error = self.compute_m_error()
        final_m_str = f", final m_err={final_m_error*100:.2f}%" if final_m_error is not None else ""
        
        if verbose:
            print("-" * 70)
            print(f"  BFGS training completed in {t_elapsed:.2f}s")
            print(f"  Final loss: {self.history['loss'][-1]:.4e}{final_m_str}")
    
    def _adam_recovery(self, n_steps: int, lr: float):
        """Run a few Adam steps to recover from BFGS failure."""
        params = list(self.net.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss, components = self.compute_loss()
            loss.backward()
            optimizer.step()
            
            # Record history
            grad_norm = sum(p.grad.norm().item()**2 for p in params if p.grad is not None)**0.5
            m_error = self.compute_m_error()
            
            self.history['loss'].append(components['loss'])
            self.history['loss_data'].append(components['loss_data'])
            self.history['loss_pde'].append(components['loss_pde'])
            self.history['loss_bc'].append(components['loss_bc'])
            self.history['loss_reg'].append(components['loss_reg'])
            self.history['grad_norm'].append(grad_norm)
            self.history['iteration'].append(self.iter_count)
            self.history['m_error'].append(m_error if m_error is not None else np.nan)
            self.iter_count += 1
    
    def _get_params_flat(self) -> np.ndarray:
        """Get all network parameters as a flat numpy array."""
        params = []
        for p in self.net.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _set_params_flat(self, params_vec: np.ndarray):
        """Set network parameters from a flat numpy array."""
        idx = 0
        for p in self.net.parameters():
            n = p.numel()
            p.data = torch.tensor(
                params_vec[idx:idx+n].reshape(p.shape),
                dtype=torch.float64,
                device=self.device
            )
            idx += n
    
    def _get_grad_flat(self) -> np.ndarray:
        """Get gradients as a flat numpy array."""
        grads = []
        for p in self.net.parameters():
            if p.grad is not None:
                grads.append(p.grad.cpu().numpy().flatten())
            else:
                grads.append(np.zeros(p.numel()))
        return np.concatenate(grads)
    
    def predict_u(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predict u at given points."""
        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float64, device=self.device)
            y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float64, device=self.device)
            u = self.forward_u(x_t, y_t)
        return u.cpu().numpy().flatten()
    
    def predict_m(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predict m at given points."""
        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float64, device=self.device)
            y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float64, device=self.device)
            m = self.forward_m(x_t, y_t)
        return m.cpu().numpy().flatten()
    
    def predict_k(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predict k = exp(m) at given points."""
        return np.exp(self.predict_m(x, y))


def plot_pinn_results(
    pinn: DarcyPINN,
    m_true_func: Callable,
    u_true: np.ndarray,
    nodes: np.ndarray,
    save_prefix: str = 'figures/pinn'
):
    """
    Plot PINN training results and compare with reference.
    
    Args:
        pinn: Trained DarcyPINN instance
        m_true_func: Function to evaluate true m(x, y)
        u_true: True u values at nodes
        nodes: (N, 2) array of node coordinates
        save_prefix: Prefix for saved figures
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    # Create evaluation grid
    n_plot = 100
    x_plot = np.linspace(0, 1, n_plot)
    y_plot = np.linspace(0, 1, n_plot)
    X, Y = np.meshgrid(x_plot, y_plot)
    X_flat, Y_flat = X.flatten(), Y.flatten()
    
    # Predict on grid
    u_pred = pinn.predict_u(X_flat, Y_flat).reshape(n_plot, n_plot)
    m_pred = pinn.predict_m(X_flat, Y_flat).reshape(n_plot, n_plot)
    k_pred = np.exp(m_pred)
    
    # True values on grid
    m_true_grid = m_true_func(X, Y)
    k_true_grid = np.exp(m_true_grid)
    u_true_grid = griddata(nodes, u_true, (X, Y), method='linear')
    
    # === Figure 1: m comparison ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmax_m = max(np.abs(m_true_grid).max(), np.abs(m_pred).max())
    
    im = axes[0].contourf(X, Y, m_true_grid, levels=20, cmap='RdBu_r',
                          vmin=-vmax_m, vmax=vmax_m)
    axes[0].set_title('Reference m†', fontsize=13)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im, ax=axes[0])
    
    im = axes[1].contourf(X, Y, m_pred, levels=20, cmap='RdBu_r',
                          vmin=-vmax_m, vmax=vmax_m)
    axes[1].set_title('PINN predicted m', fontsize=13)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im, ax=axes[1])
    
    error_m = m_pred - m_true_grid
    im = axes[2].contourf(X, Y, error_m, levels=20, cmap='seismic')
    axes[2].set_title(f'Error (max={np.abs(error_m).max():.2e})', fontsize=13)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    #plt.savefig(f'{save_prefix}_m_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === Figure 2: u comparison ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im = axes[0].contourf(X, Y, u_true_grid, levels=20, cmap='coolwarm')
    axes[0].set_title('Reference u†', fontsize=13)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im, ax=axes[0])
    
    im = axes[1].contourf(X, Y, u_pred, levels=20, cmap='coolwarm')
    axes[1].set_title('PINN predicted u', fontsize=13)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(im, ax=axes[1])
    
    error_u = u_pred - u_true_grid
    im = axes[2].contourf(X, Y, error_u, levels=20, cmap='seismic')
    axes[2].set_title(f'Error (max={np.nanmax(np.abs(error_u)):.2e})', fontsize=13)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    #plt.savefig(f'{save_prefix}_u_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # === Figure 3: Training history ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    ax = axes[0, 0]
    ax.semilogy(pinn.history['loss'], 'b-', linewidth=1)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.semilogy(pinn.history['grad_norm'], 'r-', linewidth=1)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('||∇L||', fontsize=12)
    ax.set_title('Gradient Norm', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    m_errors = np.array(pinn.history['m_error'])
    valid_idx = ~np.isnan(m_errors)
    if np.any(valid_idx):
        ax.semilogy(np.arange(len(m_errors))[valid_idx], m_errors[valid_idx] * 100, 'g-', linewidth=1)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('m error (%)', fontsize=12)
    ax.set_title('m Relative Error', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.semilogy(pinn.history['loss_data'], label='Data', linewidth=1)
    ax.semilogy(pinn.history['loss_pde'], label='PDE', linewidth=1)
    ax.semilogy(pinn.history['loss_bc'], label='BC', linewidth=1)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss Component', fontsize=12)
    ax.set_title('Loss Components', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.semilogy(pinn.history['loss_reg'], 'g-', linewidth=1, label='Reg')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Regularization', fontsize=12)
    ax.set_title(f'Regularization (β={pinn.beta:.0e})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot observation points
    ax = axes[1, 2]
    ax.scatter(pinn.x_data.cpu().numpy(), pinn.y_data.cpu().numpy(), 
               c='red', s=30, marker='x', label='Observations')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Observation Points ({pinn.n_data} points)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig(f'{save_prefix}_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute errors
    rel_error_m = np.linalg.norm(m_pred - m_true_grid) / np.linalg.norm(m_true_grid)
    
    u_pred_nodes = pinn.predict_u(nodes[:, 0], nodes[:, 1])
    rel_error_u = np.linalg.norm(u_pred_nodes - u_true) / np.linalg.norm(u_true)
    
    print(f"\nPINN Reconstruction Errors:")
    print(f"  Relative m error: {rel_error_m*100:.2f}%")
    print(f"  Relative u error: {rel_error_u*100:.4f}%")
    
    return rel_error_m, rel_error_u


def save_pinn_results(
    pinn: DarcyPINN,
    m_true_func: Callable,
    u_true: np.ndarray,
    nodes: np.ndarray,
    filepath: str = 'history/pinn_results.npz',
    extra_data: dict = None
):
    """Save PINN results to npz file.
    
    Args:
        pinn: Trained DarcyPINN instance
        m_true_func: Function for true m(x,y)
        u_true: True solution at nodes
        nodes: FEM node coordinates
        filepath: Output file path
        extra_data: Additional data to save (e.g., timing info)
    """
    # Create evaluation grid
    n_plot = 100
    x_plot = np.linspace(0, 1, n_plot)
    y_plot = np.linspace(0, 1, n_plot)
    X, Y = np.meshgrid(x_plot, y_plot)
    X_flat, Y_flat = X.flatten(), Y.flatten()
    
    # Predict on grid
    u_pred_grid = pinn.predict_u(X_flat, Y_flat).reshape(n_plot, n_plot)
    m_pred_grid = pinn.predict_m(X_flat, Y_flat).reshape(n_plot, n_plot)
    
    # True values on grid
    m_true_grid = m_true_func(X, Y)
    
    # Predictions at nodes
    u_pred_nodes = pinn.predict_u(nodes[:, 0], nodes[:, 1])
    m_pred_nodes = pinn.predict_m(nodes[:, 0], nodes[:, 1])
    
    # Build save dictionary
    save_dict = {
        # Grid data
        'X': X, 'Y': Y,
        'm_pred_grid': m_pred_grid,
        'u_pred_grid': u_pred_grid,
        'm_true_grid': m_true_grid,
        # Node data
        'nodes': nodes,
        'u_true': u_true,
        'u_pred_nodes': u_pred_nodes,
        'm_pred_nodes': m_pred_nodes,
        # Observation data
        'obs_points': np.column_stack([pinn.x_data.cpu().numpy(), 
                                       pinn.y_data.cpu().numpy()]),
        'obs_values': pinn.u_obs.cpu().numpy().flatten(),
        # Training history
        'history_loss': np.array(pinn.history['loss']),
        'history_loss_data': np.array(pinn.history['loss_data']),
        'history_loss_pde': np.array(pinn.history['loss_pde']),
        'history_loss_bc': np.array(pinn.history['loss_bc']),
        'history_loss_reg': np.array(pinn.history['loss_reg']),
        'history_grad_norm': np.array(pinn.history['grad_norm']),
        'history_m_error': np.array(pinn.history['m_error']),
        # Parameters
        'beta': pinn.beta,
        'n_obs': pinn.n_data
    }
    
    # Add extra data (e.g., timing info)
    if extra_data is not None:
        save_dict.update(extra_data)
    
    np.savez(filepath, **save_dict)
    print(f"PINN results saved to {filepath}")
