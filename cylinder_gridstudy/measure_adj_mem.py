"""Peak memory of one cylinder adjoint evaluation.

Table 2 marked this entry with an asterisk because it was never instrumented; the
number shown was the analytic trajectory storage N_u N_t x 8 B rather than a
measurement. Peak memory is set by the cached forward trajectory, so a single
objective evaluation -- one forward sweep plus its backward pass -- reaches the
same high-water mark as the full inversion, at a fraction of the cost.
"""
import os, sys, resource, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "cylinder"))
from cylinder_config import CylinderRunConfig
from cylinder_api import CylinderAPI

cfg = CylinderRunConfig()
h = os.path.join(REPO, "cylinder", "history")
cfg.saturated_path = os.path.join(h, "saturated_of.npz")
cfg.obs_path = os.path.join(h, "probe_obs_of.npz")
api = CylinderAPI(cfg)
s = api._adjoint_obs(verbose=False)
adj, u0, steps, obs = s["adj"], s["u0"], s["obs_steps"], s["obs_data"]
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
J, g, info = adj.value_and_grad_obs(np.log(cfg.nu_true), u0, steps, obs,
                                    s["Pop"], cfg.dt)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"peak host RSS after one forward+backward sweep: {peak:.3f} GB "
      f"(setup alone {before:.3f} GB)")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "cyl_adj_memory.npz"), peak_gb=peak, setup_gb=before)
