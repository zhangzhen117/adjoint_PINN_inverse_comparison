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


class PlateauStopper:
    """Run a first-order optimizer until the loss genuinely saturates.

    A fixed wall-clock cap answers "who is best on this budget", not "who converges
    where" -- and every capped run has to be reported as non-converged, which is a
    weak answer to R3.7. This stopper instead runs until the objective stops moving,
    however long that takes. Adam in particular can need on the order of 1e6 steps.

    Stopping, in order of precedence:

    1. **Learning-rate floor.** Used together with ``ReduceLROnPlateau``: once the
       schedule has decayed the rate below ``lr_min_ratio`` of its initial value,
       further steps cannot move the iterate meaningfully.
    2. **No improvement.** The best loss so far has not improved by more than
       ``rel_improve`` (relative) for ``patience_steps`` consecutive steps.
    3. **Safety cap.** A generous wall-clock ceiling, present only so a pathological
       run cannot occupy a node indefinitely. Reaching it is reported, not hidden.

    Unlike a budget cap, hitting 1 or 2 counts as convergence.
    """

    def __init__(self, lr0: float, lr_min_ratio: float = 1e-5,
                 patience_steps: int = 50_000, rel_improve: float = 1e-6,
                 max_seconds: float | None = None):
        self.lr0 = float(lr0)
        self.lr_min = float(lr0) * float(lr_min_ratio)
        self.patience_steps = int(patience_steps)
        self.rel_improve = float(rel_improve)
        self.max_seconds = max_seconds

        self.best = float("inf")
        self.best_step = 0
        self.n_steps = 0
        self._t0 = time.perf_counter()
        self.stop_reason: str | None = None

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def converged(self) -> bool:
        """True for a genuine plateau; False when the safety cap intervened."""
        return self.stop_reason in ("lr_floor", "no_improvement")

    def update(self, loss: float, lr: float) -> bool:
        self.n_steps += 1
        loss = float(loss)

        if loss < self.best * (1.0 - self.rel_improve):
            self.best = min(self.best, loss)
            self.best_step = self.n_steps
        elif loss < self.best:
            self.best = loss          # tiny improvement: track it, do not reset patience

        if lr <= self.lr_min:
            self.stop_reason = "lr_floor"
            return True
        if self.n_steps - self.best_step >= self.patience_steps:
            self.stop_reason = "no_improvement"
            return True
        if self.max_seconds is not None and self.elapsed_s >= self.max_seconds:
            self.stop_reason = "safety_cap"
            return True
        return False


def saturation_onset(times, values, tol: float = 0.01):
    """Locate where a monotone-ish descent curve flattens out.

    Running to full convergence is the right protocol, but the tail of a
    first-order run can spend most of its wall-clock buying the last fraction of a
    percent. The useful quantity to report alongside the converged value is
    therefore *when the curve effectively arrived*: the earliest point whose
    best-so-far value is already within ``tol`` (relative) of the final best.

    Returns ``(index, time, value)``, or ``(None, None, None)`` for an empty input.
    Operates on the running minimum, so noise in the raw trace cannot pull the
    onset later than it should be.
    """
    if values is None or len(values) == 0:
        return None, None, None

    running_best = []
    b = float("inf")
    for v in values:
        b = min(b, float(v))
        running_best.append(b)

    final = running_best[-1]
    # Scale-free threshold; `final` can legitimately be ~0.
    target = final * (1.0 + tol) if final > 0 else final + tol

    for i, b in enumerate(running_best):
        if b <= target:
            return i, (times[i] if times is not None else None), values[i]
    return len(values) - 1, (times[-1] if times is not None else None), values[-1]
