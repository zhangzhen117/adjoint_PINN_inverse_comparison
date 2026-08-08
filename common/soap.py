"""SOAP optimizer -- vendored reference implementation.

Source:  https://github.com/nikhilvyas/SOAP  (``soap.py``), MIT licensed.
Paper:   "SOAP: Improving and Stabilizing Shampoo using Adam", arXiv:2409.11321.

Vendored rather than pip-installed so that the optimizer used in the paper is
pinned, auditable, and identical on every node, with no dependency on a PyPI
package whose contents could differ from the reference.

SOAP runs Adam inside the eigenbasis of Shampoo's preconditioner: it accumulates
Kronecker-factored second-moment matrices ``GG`` for each tensor dimension, keeps
their eigenvectors ``Q`` (refreshed every ``precondition_frequency`` steps by one
round of power iteration plus QR), projects the gradient into that basis, takes an
Adam step there, and projects back.

--------------------------------------------------------------------------------
MODIFICATIONS FROM THE REFERENCE (all marked ``# [MOD]`` inline)

1. **Working precision is preserved.** The reference casts the preconditioner to
   ``float32`` before ``eigh``/``qr`` and casts the result back. This paper
   controls arithmetic precision across methods -- every benchmark sets
   ``torch.set_default_dtype(torch.float64)`` and the manuscript lists precision
   among the matched factors -- so silently doing the eigendecomposition in single
   precision would break that control. ``precision="preserve"`` (the default) keeps
   the parameter dtype throughout; ``precision="float32"`` reproduces the reference
   behaviour exactly, for comparison.

2. **Preconditioner buffers inherit the gradient dtype.** ``torch.zeros(sh, sh)``
   in the reference picks up the global default dtype; here it is pinned to
   ``grad.dtype`` so the buffers cannot silently disagree with the gradients.

3. ``weight_decay`` defaults to **0.0** instead of 0.01. Decoupled weight decay is
   an implicit regularizer, and the regularization functional is one of the terms
   held identical between the adjoint and the PINN formulations. A nonzero default
   would quietly add a penalty to only one side of the comparison.

Otherwise the algorithm, the argument names, and the remaining defaults are as
published.
"""

from itertools import chain

import torch
import torch.optim as optim

__all__ = ["SOAP"]


class SOAP(optim.Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params: iterable of parameters or parameter-group dicts.
        lr (float, default 3e-3): learning rate.
        betas (Tuple[float, float], default (0.95, 0.95)): Adam's (b1, b2).
        shampoo_beta (float, default -1): if >= 0, use this beta for the
            preconditioner moving average instead of betas[1].
        eps (float, default 1e-8): Adam epsilon.
        weight_decay (float, default 0.0): decoupled weight decay.  # [MOD] was 0.01
        precondition_frequency (int, default 10): eigenbasis refresh interval.
        max_precond_dim (int, default 10000): dimensions larger than this are not
            preconditioned.
        merge_dims (bool, default False): merge dimensions up to max_precond_dim.
        precondition_1d (bool, default False): precondition 1D tensors (biases).
        normalize_grads (bool, default False): normalize the update per layer.
        data_format (str, default "channels_first"): layout for 4D conv weights.
        correct_bias (bool, default True): Adam bias correction.
        precision (str, default "preserve"): "preserve" keeps the parameter dtype
            in the eigendecomposition; "float32" reproduces the reference.  # [MOD]
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.0,          # [MOD] reference default is 0.01
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        precision: str = "preserve",        # [MOD] new
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format
        if precision not in ("preserve", "float32"):
            raise ValueError('precision must be "preserve" or "float32"')
        self._precision = precision            # [MOD]

    # [MOD] helper: the dtype the eigendecomposition runs in.
    def _work_dtype(self, dtype):
        return dtype if self._precision == "preserve" else torch.float32

    def merge_dims(self, grad, max_precond_dim):
        """Merge dimensions until their product exceeds max_precond_dim."""
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []

        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        return grad.reshape(new_shape)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if "Q" not in state:
                    self.init_preconditioner(
                        grad, state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(group["shampoo_beta"] if group["shampoo_beta"] >= 0
                                      else group["betas"][1]),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self.update_preconditioner(
                        grad, state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"])
                    # First step is skipped so the current gradient is never used
                    # in its own projection.
                    continue

                grad_projected = self.project(
                    grad, state, merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"])

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                exp_avg_projected = exp_avg

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                norm_grad = self.project_back(
                    exp_avg_projected / denom, state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"])

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                # Decoupled weight decay (AdamW-style), applied after the step.
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Updated after the step, so current gradients never enter their
                # own projection basis.
                self.update_preconditioner(
                    grad, state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"])

        return loss

    def init_preconditioner(self, grad, state, precondition_frequency=10,
                            shampoo_beta=0.95, max_precond_dim=10000,
                            precondition_1d=False, merge_dims=False):
        """Initialize the preconditioner matrices (L and R in the paper)."""
        state["GG"] = []
        # [MOD] buffers inherit the gradient dtype rather than the global default.
        zeros = lambda n: torch.zeros(n, n, device=grad.device, dtype=grad.dtype)

        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(zeros(grad.shape[0]))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                state["GG"].append([] if sh > max_precond_dim else zeros(sh))

        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Project the gradient into the eigenbasis of the preconditioner."""
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(self, grad, state, max_precond_dim=10000,
                              merge_dims=False, precondition_1d=False):
        """Update the preconditioner matrices and their eigenbases."""
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(
                state["exp_avg"], state, merge_dims=merge_dims,
                max_precond_dim=max_precond_dim)

        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0),
                                     1 - state["shampoo_beta"])
        else:
            new_grad = self.merge_dims(grad, max_precond_dim) if merge_dims else grad
            for idx, sh in enumerate(new_grad.shape):
                if sh <= max_precond_dim:
                    outer_product = torch.tensordot(
                        new_grad, new_grad,
                        # Contract over every dimension except idx.
                        dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                    )
                    state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)

        if state["step"] > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"], state, merge_dims=merge_dims,
                max_precond_dim=max_precond_dim)

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Project the gradient back into the original space."""
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        """Eigenbases of the preconditioner via ``torch.linalg.eigh``."""
        final = []
        for m in mat:
            if len(m) == 0:
                final.append([])
                continue
            # [MOD] run in the parameter dtype unless precision="float32".
            original_type, original_device = m.data.dtype, m.data.device
            work = self._work_dtype(original_type)
            mw = m.data.to(work)
            eye = torch.eye(mw.shape[0], device=mw.device, dtype=work)
            try:
                _, Q = torch.linalg.eigh(mw + 1e-30 * eye)
            except Exception:
                _, Q = torch.linalg.eigh(
                    mw.to(torch.float64)
                    + 1e-30 * eye.to(torch.float64))
                Q = Q.to(work)
            Q = torch.flip(Q, [1])
            final.append(Q.to(device=original_device, dtype=original_type))
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """Eigenbases via one round of power iteration followed by QR."""
        precond_list = state["GG"]
        orth_list = state["Q"]

        orig_shape = state["exp_avg_sq"].shape
        permuted_shape = None
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
        exp_avg_sq = (self.merge_dims(state["exp_avg_sq"], max_precond_dim)
                      if merge_dims else state["exp_avg_sq"])

        final = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if len(m) == 0:
                final.append([])
                continue
            # [MOD] run in the parameter dtype unless precision="float32".
            original_type, original_device = m.data.dtype, m.data.device
            work = self._work_dtype(original_type)
            mw, ow = m.data.to(work), o.data.to(work)

            est_eig = torch.diag(ow.T @ mw @ ow)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            ow = ow[:, sort_idx]
            power_iter = mw @ ow
            Q, _ = torch.linalg.qr(power_iter)
            final.append(Q.to(device=original_device, dtype=original_type))

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state["exp_avg_sq"] = exp_avg_sq
        return final
