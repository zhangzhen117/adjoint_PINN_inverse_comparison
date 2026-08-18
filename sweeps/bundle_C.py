"""Bundle C -- does a modern PINN setup change the cylinder result? (R1.5, R3.4)

R1.5 notes that each benchmark uses a single architecture and R3.4 asks whether the
conclusions are architecture-dependent, naming Fourier features and adaptive
activations. Rather than sweep width and depth -- which on this benchmark mostly
measures how fragile the training setup is -- bundle C compares three setups at the
paper's network size:

1. **paper** -- the published run: plain tanh MLP 32x3, Adam warmup then SSBroyden.
   This is the eps_nu = 5.24e-3 in Table 2.
2. **modern** -- every technique from the jaxpi2 recipe at once
   (github.com/sifanexisted/jaxpi2, and the author's cylinder variant
   ``cyl_pinn_pt.py``): gated ModifiedMlp + SOAP + gradient-norm loss balancing +
   pseudo-time relaxation of the residual.
3. **vanilla_adam** -- the paper's own plain tanh MLP trained with Adam alone.
   Included to show that a vanilla PINN does *not* solve this problem, so the
   comparison in the paper is against a competent PINN rather than a straw man.
4. **paper_converged** -- the published algorithm run to a plateau rather than to
   its fixed 10 restarts. The published run's loss is still descending when it
   stops, so it is a fixed-budget result; without this arm the converged modern
   setup would be compared against a truncated baseline.

The two arms that share the first-order training loop (modern, vanilla_adam) also
share its learning-rate schedule, so the decay is a matched control rather than a
difference between them.

Parameter counts are held near the paper's 2307: ModifiedMlp 32x3 is 2563, the
plain MLP is 2307. So the three arms differ in *training setup*, not capacity.

Five seeds each, since bundle D showed the cylinder PINN has real seed spread and a
single run per arm could not distinguish a technique from an initialization.

Note for the write-up: grad-norm balancing is adaptive loss weighting, whereas the
manuscript states all weights are unity (main.tex:455). That bears directly on
referee R1.6, which is assigned to the other author -- the two responses need to
agree.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cases", "case4_cylinder"))

from common.instrument import (Counters, RunTimer, require_l40s,  # noqa: E402
                               ssbroyden_hessian_bytes)
from common.seeding import set_seed                                # noqa: E402
from common.sweep import make_record                               # noqa: E402
from sweeps._runner import cli                                     # noqa: E402

SEEDS = (0, 1, 2, 3, 4)

# (setup, arch, optimizer, gradnorm, pseudotime)
# NOTE: rows() is addressed positionally by the SLURM array, and a running array
# task resolves its index against this file at execution time. New cases must
# therefore be APPENDED, never inserted -- inserting renumbers every later row and
# silently repoints in-flight jobs at the wrong configuration.
CASES = {
    "paper":           ("paper",        "plain",    "ssbroyden2", False, False),
    "modern":          ("modern",       "modified", "soap",       True,  True),
    "vanilla_adam":    ("vanilla_adam", "plain",    "adam",       False, False),
    "paper_converged": ("paper",        "plain",    "ssbroyden2", False, False),
}


def rows():
    return [{"case": c, "seed": s} for c in CASES for s in SEEDS]


def run_one(row, index):
    require_l40s()

    from cylinder_config import CylinderRunConfig
    import cylinder_pinn_inverse as cpi

    case = row["case"]
    setup, arch, optimizer, gradnorm, pseudotime = CASES[case]

    run_dir = os.path.join(REPO, "results", "C_runs", f"{case}_s{row['seed']}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = CylinderRunConfig()
    cfg.pinn_setup = setup
    cfg.jaxpi_arch = arch
    cfg.jaxpi_optimizer = "adam" if optimizer == "adam" else "soap"
    cfg.jaxpi_gradnorm = gradnorm
    cfg.jaxpi_pseudotime = pseudotime
    cfg.seed_inv = 44 + row["seed"]

    if case == "paper_converged":
        # Same algorithm as the published run, but restarted until the loss stops
        # improving instead of stopping at a fixed 10 restarts. The published arm's
        # loss is still descending when it halts, so without this the comparison
        # against the converged modern arm would be against a truncated baseline.
        cfg.n_restarts_inv = 60
        cfg.inv_restart_plateau = 5

    # The cylinder's default paths are relative and resolve against the CWD, which
    # for a sweep is the repo root. Point the inputs at the cached warmup and
    # observation files -- the 807 s saturation run is reused, never recomputed.
    cyl_hist = os.path.join(REPO, "cases", "case4_cylinder", "history")
    cfg.saturated_path = os.path.join(cyl_hist, "saturated.npz")
    cfg.obs_path = os.path.join(cyl_hist, "probe_obs.npz")
    cfg.hist_dir = run_dir
    cfg.fig_dir = fig_dir
    cfg.pinn_inv_npz = os.path.join(run_dir, "pinn_inv.npz")
    cfg.pinn_inv_model = os.path.join(run_dir, "pinn_inv_model.pt")

    set_seed(cfg.seed_inv)
    counters = Counters()

    with RunTimer() as timer:
        out = cpi.train(cfg)

    # Settling diagnostic: how much does nu still move over the last 10% of the
    # recorded trajectory? A converged run should have this near zero. Also record
    # the oracle-best, purely so the gap between "reachable" and "selectable" is
    # visible in the data -- it is NOT a reportable accuracy.
    nu_hist = np.asarray(out["hist"]["nu"], dtype=float)
    nu_true = float(out["nu_true"])
    e_hist = np.abs(nu_hist - nu_true) / nu_true
    tail = e_hist[int(0.9 * len(e_hist)):] if len(e_hist) > 10 else e_hist
    settle_band = float(tail.max() - tail.min()) if len(tail) else float("nan")
    oracle_best = float(e_hist.min()) if len(e_hist) else float("nan")

    n_params = sum(p.numel() for p in out["model"].parameters())
    n_evals = len(out["hist"]["loss"])
    counters.n_params = n_params
    counters.hessian_bytes_analytic = (ssbroyden_hessian_bytes(n_params)
                                       if optimizer == "ssbroyden2" else 0)
    counters.n_fev = counters.n_iter = n_evals
    counters.residual(cfg.n_pde * max(n_evals, 1))

    return make_record(
        bundle="C", benchmark="cylinder", method="pinn", representation="scalar",
        optimizer=optimizer if case != "modern" else "soap+gradnorm+pts",
        arch=f"{case}-w{cfg.hidden}d{cfg.layers}",
        seed=cfg.seed_inv, noise=cfg.obs_noise, gamma=0.0, beta=0.0,
        # The unknown is the scalar nu, so eps_f is its relative error.
        eps_f=float(out["rel_err"]), eps_u=None,
        converged=None, stop_reason="maxiter",
        **counters.as_dict(), **timer.as_dict(),
        case=case, nu_rec=float(out["nu_rec"]), nu_true=nu_true,
        restart_stop=out.get("restart_stop"), n_restarts_run=out.get("n_restarts_run"),
        settle_band=settle_band, oracle_best_eps=oracle_best,
        gradnorm=out.get("gradnorm"), pseudotime=out.get("pseudotime"),
        cfg_snapshot=cfg.snapshot(),
    )


if __name__ == "__main__":
    cli("C", rows, run_one)
