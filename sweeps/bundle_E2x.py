"""Bundle E2x -- extend the gamma grid so the optimum is bracketed.

E2 found the adjoint's error still falling at the largest gamma tested (1e-2) at
both noise levels, so "each method reported at its own best gamma" would be resting
on a boundary value rather than a minimum. This adds gamma in {3e-2, 1e-1} at the
same two noise levels, which brackets it.

Records append to results/E.jsonl and carry sub="E2" so they merge with the
original grid at analysis time.
"""

from __future__ import annotations

import sys

from sweeps._runner import cli
from sweeps.bundle_E import run_one, SEEDS

GAMMAS_EXT = (3e-2, 1e-1)
GAMMA_NOISE = (0.01, 0.05)


def rows():
    return [{"sub": "E2", "method": m, "noise": n, "gamma": g, "seed": s}
            for n in GAMMA_NOISE for g in GAMMAS_EXT
            for m in ("adjoint", "pinn") for s in SEEDS]


if __name__ == "__main__":
    # Same results file as bundle E; sub="E2" keeps the two halves of the grid
    # together for analysis.
    sys.argv += ["--out", "results/E.jsonl"]
    cli("E2x", rows, run_one)
