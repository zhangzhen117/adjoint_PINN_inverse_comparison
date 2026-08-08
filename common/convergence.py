"""Shared stopping rule for the optimizer ablation.

The optimizer comparison (referee R3.7) puts Adam, SOAP and SSBroyden side by side.
Those have no common notion of an "iteration" -- one SSBroyden step costs orders of
magnitude more than one Adam step -- so comparing them at equal iteration counts
would flatter the cheap-step methods. Every optimizer is therefore run *to
convergence* under one rule, and the wall-clock each needs is part of the result.

Two ways to stop:

1. **Tolerance.** The objective has stopped moving: the relative spread of the loss
   over the last ``window`` evaluations is below ``rel_tol``.
2. **Wall-clock cap.** A hard ceiling, defaulting to twice the SSBroyden baseline for
   that benchmark. Without it Adam does not terminate on these problems. A run that
   hits the cap is recorded ``converged=False`` with ``stop_reason="walltime_cap"``
   and is reported as non-converged in the manuscript -- never silently truncated
   and presented as a converged result.
"""

from __future__ import annotations

import time
from collections import deque


class ConvergenceMonitor:
    """Track an objective sequence and decide when to stop.

    Parameters
    ----------
    rel_tol
        Stop once ``(max-min)/max(|mean|, tiny)`` over the trailing window drops
        below this. Relative rather than absolute because the four benchmarks'
        objectives differ by many orders of magnitude.
    window
        Number of trailing evaluations considered. 500 is long enough that a
        quasi-Newton line search stalling for a few steps does not trigger a stop.
    walltime_cap_s
        Hard ceiling in seconds. ``None`` disables it (used for the baseline runs
        that *define* the cap).
    max_evals
        Optional evaluation ceiling, as a backstop against a pathological run.
    """

    def __init__(
        self,
        rel_tol: float = 1e-8,
        window: int = 500,
        walltime_cap_s: float | None = None,
        max_evals: int | None = None,
    ) -> None:
        self.rel_tol = float(rel_tol)
        self.window = int(window)
        self.walltime_cap_s = walltime_cap_s
        self.max_evals = max_evals

        self._hist: deque[float] = deque(maxlen=self.window)
        self._t0 = time.perf_counter()
        self.n_evals = 0
        self.stop_reason: str | None = None

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def converged(self) -> bool:
        """True only for a genuine tolerance stop, not for a cap or an eval ceiling."""
        return self.stop_reason == "tolerance"

    def update(self, loss: float) -> bool:
        """Record one objective value; return True when the run should stop."""
        self.n_evals += 1
        self._hist.append(float(loss))

        if self.walltime_cap_s is not None and self.elapsed_s >= self.walltime_cap_s:
            self.stop_reason = "walltime_cap"
            return True

        if self.max_evals is not None and self.n_evals >= self.max_evals:
            self.stop_reason = "max_evals"
            return True

        if len(self._hist) == self.window:
            lo, hi = min(self._hist), max(self._hist)
            scale = max(abs(sum(self._hist) / len(self._hist)), 1e-300)
            if (hi - lo) / scale < self.rel_tol:
                self.stop_reason = "tolerance"
                return True

        return False

    def finish(self, reason: str = "maxiter") -> None:
        """Record a stop imposed from outside (e.g. the optimizer's own maxiter)."""
        if self.stop_reason is None:
            self.stop_reason = reason

    def as_dict(self) -> dict:
        return {
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "n_evals_monitored": self.n_evals,
            "elapsed_s": self.elapsed_s,
        }
