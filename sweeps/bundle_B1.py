"""Bundle B1 -- PINN optimizers at their tuned learning rate, five seeds.

Follows bundle B0, which swept the learning rate on one seed. Two things B0 showed
shape this stage:

* Run to saturation, B0's optima moved: Adam's best is 1e-2 (not 3e-3) and SOAP's
  is 3e-3 (not 1e-3). Both are interior to the grid, so the minimum is bracketed
  and no extension is needed.
* All eight B0 runs stopped on the learning-rate floor, i.e. genuinely converged
  rather than truncated at a budget, so these five-seed runs are convergence
  comparisons and the wall-clock each needs is part of the result.

The SSBroyden arm is not repeated here -- bundle A's pinn/nn rows are the identical
configuration at the same five seeds.
"""

from __future__ import annotations

from sweeps._runner import cli
from sweeps.bundle_B import run_one

SEEDS = (0, 1, 2, 3, 4)
TUNED = {"adam": 1e-2, "soap": 3e-3}   # from the converged B0 sweep


def rows():
    # No learning-rate extension needed: with the converged B0 sweep both optima
    # are interior (Adam 1e-2 between 3e-3 and 3e-2; SOAP 3e-3 between 1e-3 and
    # 1e-2), so the minimum is already bracketed on both sides.
    return [{"sub": "B1", "side": "pinn", "optimizer": opt, "lr": lr, "seed": s}
            for opt, lr in TUNED.items() for s in SEEDS]


if __name__ == "__main__":
    import sys
    sys.argv += ["--out", "results/B.jsonl"]
    cli("B1", rows, run_one)
