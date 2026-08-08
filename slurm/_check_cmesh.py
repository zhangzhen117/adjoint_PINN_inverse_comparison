"""Validate the truth-free C_mesh calibration against the original truth-based one.

If the prior-based estimate lands close to the truth-based value, the original
number was not cherry-picked and the change is a provenance fix rather than a
change of result -- which is exactly what we want to be able to state in the
response letter.
"""
import sys
import numpy as np

sys.path.insert(0, "Darcy_New")

from cfg import DarcyConfig, calibrate_C_mesh, pinn_reg_weights
from darcy_adjoint import DarcyInverse, create_observation_grid
from GRF import GaussianRF

cfg = DarcyConfig()
print("config:", {k: v for k, v in cfg.as_dict().items() if k in
                  ("nx", "ny", "element_degree", "grf_modes", "gamma", "beta",
                   "n_obs_x", "n_obs_y", "max_iter", "C_mesh_source")})

obs_points = create_observation_grid(n_x=cfg.n_obs_x, n_y=cfg.n_obs_y, margin=cfg.obs_margin)
inv = DarcyInverse(
    nx=cfg.nx, ny=cfg.ny, element_degree=cfg.element_degree,
    f_given=lambda x, y: cfg.source_value * np.ones_like(x),
    beta=cfg.beta, gamma=cfg.gamma,
    solver_type=cfg.linear_solver, obs_points=obs_points,
)
print(f"DOFs={inv.n_dofs}  elements={inv.n_elements}  obs={inv.n_obs}  "
      f"internal_edges={inv.n_internal_edges}")

RF = GaussianRF(cfg.grf_tau, cfg.grf_alpha)
rng = np.random.RandomState(cfg.truth_seed)
coefs_true = rng.randn(cfg.grf_modes)
m_true = RF.sample(inv.element_centroids, coefs_true).flatten()
print(f"m_true: min={m_true.min():.4f} max={m_true.max():.4f} std={m_true.std():.4f}")

C_prior, diag = calibrate_C_mesh(inv, RF, cfg, coefs_true=coefs_true)
print("\n--- C_mesh from PRIOR samples (new default, truth-free) ---")
for k, v in diag.items():
    print(f"  {k:20s} {v}")

cfg_truth = DarcyConfig(C_mesh_source="truth")
C_truth, diag_t = calibrate_C_mesh(inv, RF, cfg_truth, coefs_true=coefs_true)
print("\n--- C_mesh from TRUTH (original notebook behaviour) ---")
for k, v in diag_t.items():
    print(f"  {k:20s} {v}")

print("\n--- comparison ---")
print(f"  C_mesh(prior) = {C_prior:.6f}")
print(f"  C_mesh(truth) = {C_truth:.6f}")
print(f"  relative difference = {abs(C_prior - C_truth) / C_truth * 100:.2f}%")

bp, gp_prior = pinn_reg_weights(cfg, C_prior)
_, gp_truth = pinn_reg_weights(cfg, C_truth)
print(f"\n  gamma_pinn(prior) = {gp_prior:.6e}")
print(f"  gamma_pinn(truth) = {gp_truth:.6e}")
print(f"  beta_pinn         = {bp:.3e}")
print(f"  relative difference in gamma_pinn = "
      f"{abs(gp_prior - gp_truth) / gp_truth * 100:.2f}%")

# Cheap sanity check that the adjoint path still works end to end.
y_obs, u_clean = inv.generate_observations(
    m_true, noise_level=cfg.noise_level, noise_type=cfg.noise_type, seed=cfg.noise_seed)
J, grad_m, mis, reg = inv.objective_and_gradient(np.zeros(inv.n_elements), y_obs)
print(f"\nadjoint smoke: J={J:.6e} misfit={mis:.6e} reg={reg:.6e} |grad|={np.linalg.norm(grad_m):.4e}")
print("OK")
