"""Pseudo-time stepping for PINN residuals (jaxpi2 / arXiv:2604.23528).

PyTorch port of the scheme in
``cylinder_agent/jaxpi_run/cyl_pinn_pt.py`` (JAX), which follows jaxpi2. The PDE
residual is relaxed toward the solution of the previous iterate,

    R[u_theta] + theta_k * ( u_k(theta) - u_k(theta_prev) )

so training behaves like an implicit pseudo-time march: each step solves a problem
anchored to the previous state rather than the raw steady residual, which damps the
stiff early transient that otherwise sends a cold-started PINN to a trivial minimum.

Three details matter and are easy to get wrong -- an earlier version of this file
got all three wrong and the resulting cylinder run oscillated instead of converging:

1. **The previous iterate is refreshed every step**, not every few thousand. With an
   every-step update, ``(u - u_prev) * theta`` is a genuine discrete pseudo-time
   derivative; with a stale snapshot it is a stiff anchor to an old solution that
   actively fights convergence.
2. **theta = 1/tau is adaptive and per-component**, computed from the ratio of the
   change in the residual to the change in the solution between consecutive
   iterates, ``||dR_k|| / ||du_k||``. A fixed hand-chosen weight has the wrong
   scale, and the correct scale differs per equation component.
3. **A cosine shrink** damps theta as the residual drops, so the anchor fades and
   does not bias the converged solution.

theta is recomputed every ``update_every`` steps (200 in the reference) and smoothed
with momentum (0.7), because it needs two extra residual evaluations.

Inverse-problem note: the reference targets forward problems. Here only the PDE
residual is anchored -- the data-misfit term is left alone, so the observations
always pull on the true solution rather than on the previous iterate.
"""

from __future__ import annotations

import copy
import math

import torch

__all__ = ["PseudoTimeStepper"]


class PseudoTimeStepper:
    """Adaptive per-component pseudo-time weights with an every-step prev iterate.

    Parameters
    ----------
    n_components
        Number of residual components carrying their own theta (u, v, p -> 3).
    update_every
        Steps between recomputations of theta. The previous *iterate* is still
        refreshed every step; this is only the cadence of the weight estimate.
    momentum
        Smoothing of theta across updates (0.7 in the reference).
    start_log_drop, end_log_drop, min_factor
        Cosine shrink: the factor goes from 1 to ``min_factor`` as
        ``log10(loss0 / loss)`` moves from ``start_log_drop`` to ``end_log_drop``.
    clip
        Bounds on theta, as in the reference (1e-2, 100).
    """

    def __init__(self, n_components: int = 2, update_every: int = 200,
                 momentum: float = 0.7, start_log_drop: float = 2.0,
                 end_log_drop: float = 6.0, min_factor: float = 0.1,
                 clip: tuple[float, float] = (1e-2, 100.0)):
        self.n = int(n_components)
        self.update_every = int(update_every)
        self.momentum = float(momentum)
        self.start_log_drop = float(start_log_drop)
        self.end_log_drop = float(end_log_drop)
        self.min_factor = float(min_factor)
        self.clip = clip

        self.theta = [0.0] * self.n          # off until the first update
        self.losses0: list[float] | None = None
        self.prev = None
        self.n_updates = 0

    # -- previous iterate -----------------------------------------------------
    def refresh(self, model) -> None:
        """Snapshot the current parameters. Call every step, after the update."""
        if self.prev is None:
            self.prev = copy.deepcopy(model)
        else:
            with torch.no_grad():
                for q, p in zip(self.prev.parameters(), model.parameters()):
                    q.copy_(p)
        for q in self.prev.parameters():
            q.requires_grad_(False)

    @property
    def active(self) -> bool:
        return self.prev is not None and any(t > 0 for t in self.theta)

    def should_update(self, step: int) -> bool:
        return self.prev is not None and step % self.update_every == 0

    # -- adaptive weights ------------------------------------------------------
    def update_theta(self, res_now, res_prev, sol_now, sol_prev,
                     losses: list[float]) -> list[float]:
        """theta_k = ||dR_k|| / ||du_k|| * shrink, smoothed with momentum.

        All arguments are per-component sequences of detached tensors over the same
        collocation batch.
        """
        if self.losses0 is None:
            self.losses0 = [float(l) for l in losses]

        new = []
        for k in range(self.n):
            dR = (res_now[k] - res_prev[k]).detach()
            du = (sol_now[k] - sol_prev[k]).detach()
            ratio = float(torch.linalg.norm(dR)) / (float(torch.linalg.norm(du)) + 1e-8)

            log_drop = math.log10((self.losses0[k] + 1e-8) / (float(losses[k]) + 1e-8))
            pgr = min(max((log_drop - self.start_log_drop)
                          / (self.end_log_drop - self.start_log_drop), 0.0), 1.0)
            factor = self.min_factor + (1.0 - self.min_factor) * 0.5 * (1.0 + math.cos(math.pi * pgr))

            t = min(max(ratio * factor, self.clip[0]), self.clip[1])
            m = self.momentum if self.n_updates > 0 else 0.0
            new.append(m * self.theta[k] + (1.0 - m) * t)

        self.theta = new
        self.n_updates += 1
        return list(self.theta)

    def state(self) -> dict:
        return {"pts_theta": list(self.theta), "pts_updates": self.n_updates,
                "pts_active": self.active}
