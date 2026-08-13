"""Architectures from the jaxpi2 reference setup, reimplemented in PyTorch.

Source of the design: https://github.com/sifanexisted/jaxpi2 (``jaxpi/archs.py``).
That codebase is jax/flax, so these are reimplementations rather than a port; the
algebra follows the reference exactly and the defaults follow its ``bfs_flow``
example, which is the closest published configuration to our cylinder problem.

Used by the bundle-C architecture ablation. The point of running the sweep under
this setup as well as the paper's plain tanh MLP is that under the paper's setup
only the production 32x3 network recovers the viscosity at all -- every other
width/depth lands at eps_nu ~ 0.3-1.4 -- which measures the fragility of the
training setup more than it measures architecture. A stronger baseline is what
makes the architecture question answerable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["AdaptiveActivation", "FourierEmbedding", "ModifiedMlp", "PlainMlp",
           "get_activation"]


def get_activation(name: str) -> nn.Module:
    acts = {"tanh": nn.Tanh, "swish": nn.SiLU, "silu": nn.SiLU,
            "gelu": nn.GELU, "relu": nn.ReLU, "sin": None}
    if name.lower() == "sin":
        class Sin(nn.Module):
            def forward(self, x):
                return torch.sin(x)
        return Sin()
    cls = acts.get(name.lower())
    if cls is None:
        raise ValueError(f"unknown activation {name!r}")
    return cls()


class FourierEmbedding(nn.Module):
    """Random Fourier features, ``[sin(Bx), cos(Bx)]`` with ``B ~ N(0, scale^2)``.

    Matches ``jaxpi.archs.FourierEmbs``. ``B`` is drawn once and held fixed as a
    buffer, so it is part of the seeded initialization and moves with the model.
    Output dimension is ``2 * embed_dim``.
    """

    def __init__(self, in_dim: int, embed_dim: int = 256, embed_scale: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, embed_dim) * embed_scale)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ModifiedMlp(nn.Module):
    """The gated "modified MLP" of Wang et al., as in ``jaxpi.archs.ModifiedMlp``.

    Two encoders are computed once from the input::

        U = act(W_u x + b_u),    V = act(W_v x + b_v)

    and every hidden layer then mixes its output against them::

        x = act(W x + b)
        x = x * U + (1 - x) * V

    The gate keeps a path from the input to every depth, which is what makes the
    architecture markedly easier to train than a plain MLP of the same size on
    stiff PDE residuals.

    ``num_layers`` counts the gated hidden layers, matching the reference, so the
    total parameter count is roughly ``(num_layers + 2)`` dense blocks.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, activation: str = "swish",
                 fourier_emb: dict | None = None):
        super().__init__()
        self.embed = None
        d = in_dim
        if fourier_emb:
            self.embed = FourierEmbedding(in_dim, **fourier_emb)
            d = self.embed.out_dim

        self.act = get_activation(activation)
        self.enc_u = nn.Linear(d, hidden_dim)
        self.enc_v = nn.Linear(d, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(d if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_layers)])
        self.out = nn.Linear(hidden_dim, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)   # Glorot normal, as in flax
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        u = self.act(self.enc_u(x))
        v = self.act(self.enc_v(x))
        for lin in self.layers:
            x = self.act(lin(x))
            x = x * u + (1.0 - x) * v
        return self.out(x)


class PlainMlp(nn.Module):
    """Plain MLP with the same input-embedding option, for a like-for-like control."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 32,
                 num_layers: int = 3, activation: str = "tanh",
                 fourier_emb: dict | None = None):
        super().__init__()
        self.embed = None
        d = in_dim
        if fourier_emb:
            self.embed = FourierEmbedding(in_dim, **fourier_emb)
            d = self.embed.out_dim
        act = activation
        layers: list[nn.Module] = [nn.Linear(d, hidden_dim), get_activation(act)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), get_activation(act)]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        return self.net(x)


class AdaptiveActivation(nn.Module):
    """Layer-wise locally adaptive activation (L-LAAF).

    Jagtap, Kawaguchi and Karniadakis, *Adaptive activation functions accelerate
    convergence in deep and physics-informed neural networks*, JCP 404 (2020).

    Applies ``sigma(n * a * x)`` with a single trainable scalar ``a`` per layer and
    a fixed scale factor ``n``. Initialising ``a = 1/n`` makes the network identical
    to its fixed-activation counterpart at step zero, so any difference is
    attributable to the adaptation and not to a different starting point. The extra
    cost is one parameter per layer, which leaves the capacity comparison clean.
    """

    def __init__(self, base: str = "tanh", n: float = 10.0):
        super().__init__()
        self.act = get_activation(base)
        self.n = float(n)
        self.a = nn.Parameter(torch.tensor(1.0 / float(n)))

    def forward(self, x):
        return self.act(self.n * self.a * x)
