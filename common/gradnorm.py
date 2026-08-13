"""Gradient-norm loss balancing, as in jaxpi2's ``compute_loss_weights``.

Reference: https://github.com/sifanexisted/jaxpi2 (``jaxpi/models.py:400``), which
implements the scheme of Wang, Teng and Perdikaris. Each loss term is weighted by
how *small* its gradient is relative to the others,

    w_i = mean_j ||grad L_j||  /  ( ||grad L_i|| + eps * mean_j ||grad L_j|| )

so terms whose gradients are being drowned out get amplified. Weights are refreshed
every ``update_every`` steps and smoothed with momentum, because recomputing them
costs one backward pass per loss term.

**This is adaptive loss balancing, and the manuscript currently states that all loss
weights are unity and that no manual balancing is performed** (main.tex:455, 1157).
Referee comment R1.6 asks precisely whether the unit weights were motivated by
preliminary experiments. Anything measured with this class is therefore evidence
*for* answering R1.6, not a silent departure from the stated protocol, and the two
must be described consistently in the response.
"""

from __future__ import annotations

import torch

__all__ = ["GradNormWeighter"]


class GradNormWeighter:
    """Maintain per-term loss weights from their gradient norms.

    Parameters
    ----------
    keys
        Names of the loss terms, fixed for the run.
    params
        The parameters the gradients are taken with respect to.
    update_every
        Recompute every this many optimizer steps. The reference uses 1000; each
        update costs one backward pass per term, so this is not free.
    momentum
        Exponential smoothing of the weights, as in the reference (0.9).
    eps
        Guard in the denominator, relative to the mean gradient norm.
    """

    def __init__(self, keys, params, update_every: int = 1000,
                 momentum: float = 0.9, eps: float = 1e-5):
        self.keys = list(keys)
        self.params = [p for p in params if p.requires_grad]
        self.update_every = int(update_every)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.weights = {k: 1.0 for k in self.keys}
        self.last_norms: dict[str, float] = {}
        self._step = 0

    def should_update(self, step: int) -> bool:
        return self.update_every > 0 and step % self.update_every == 0

    @torch.no_grad()
    def _apply(self, norms: dict[str, float]) -> None:
        vals = [v for v in norms.values() if v > 0 and torch.isfinite(torch.tensor(v))]
        if not vals:
            return
        mean_norm = sum(vals) / len(vals)
        for k in self.keys:
            n = norms.get(k, 0.0)
            target = mean_norm / (n + self.eps * mean_norm) if n > 0 else 1.0
            m = self.momentum
            self.weights[k] = m * self.weights[k] + (1.0 - m) * float(target)
        self.last_norms = norms

    def update(self, loss_terms: dict[str, "torch.Tensor"]) -> dict[str, float]:
        """Recompute the weights from the current per-term gradient norms.

        ``loss_terms`` must be the differentiable tensors, not floats. The graph is
        retained across terms, so build them from one forward pass and call this
        before the optimizer step.
        """
        norms: dict[str, float] = {}
        for i, k in enumerate(self.keys):
            L = loss_terms.get(k)
            if L is None or not L.requires_grad:
                norms[k] = 0.0
                continue
            g = torch.autograd.grad(
                L, self.params,
                retain_graph=True,          # later terms still need the graph
                create_graph=False,
                allow_unused=True,
            )
            flat = torch.cat([t.reshape(-1) for t in g if t is not None]) \
                if any(t is not None for t in g) else None
            norms[k] = float(torch.linalg.norm(flat)) if flat is not None else 0.0
        self._apply(norms)
        return dict(self.weights)

    def weighted_total(self, loss_terms: dict[str, "torch.Tensor"]):
        """Combine the terms with the current weights (weights held constant)."""
        total = None
        for k in self.keys:
            L = loss_terms.get(k)
            if L is None:
                continue
            term = self.weights[k] * L
            total = term if total is None else total + term
        return total

    def state(self) -> dict:
        return {"weights": dict(self.weights), "grad_norms": dict(self.last_norms)}
