"""Bundle E2y -- complete the gamma sweep at the remaining noise levels.

The gamma grid was only run at sigma = 1% and 5%. Since gamma must be selected per
noise level -- and, crucially, the *same* gamma must then be used by both methods,
because matched regularization is one of the controls the paper claims -- the grid
has to cover every noise level, not just two.

Already covered:
  sigma in {1%, 5%}   x gamma in {0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1}
  all sigma           x gamma = 1e-3  (from E1)

Added here:
  sigma in {0, 0.1%, 10%} x gamma in {0, 1e-4, 1e-2, 3e-2, 1e-1}

EKI is excluded: it carries no explicit gamma, its regularization coming from the
ensemble prior instead, so a gamma axis is meaningless for it.

Records append to results/E.jsonl with sub="E2" so the whole grid analyses together.
"""

from __future__ import annotations

import sys

from sweeps._runner import cli
from sweeps.bundle_E import run_one, SEEDS

# 1e-3 is omitted at every level because E1 already ran it there.
GAMMAS = (0.0, 1e-4, 1e-2, 3e-2, 1e-1)
NOISE = (0.0, 0.001, 0.10)


def rows():
    return [{"sub": "E2", "method": m, "noise": n, "gamma": g, "seed": s}
            for n in NOISE for g in GAMMAS
            for m in ("adjoint", "pinn") for s in SEEDS]


if __name__ == "__main__":
    sys.argv += ["--out", "results/E.jsonl"]
    cli("E2y", rows, run_one)
