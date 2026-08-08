"""Single seeding entry point for every benchmark.

Referees R1.7 and R3.5 ask for multi-initialization statistics. Before this module
the seeding was inconsistent across benchmarks -- the Burgers notebook never called
``torch.manual_seed`` at all, so its network initialization was not reproducible.
Every adjoint and PINN entry point now calls :func:`set_seed` first, and the seed is
recorded in the result row.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and CUDA) from one integer.

    Parameters
    ----------
    seed
        The seed. Also written to ``PYTHONHASHSEED`` so that any subprocess
        inherits it.
    deterministic
        If True, additionally force cuDNN into deterministic mode. This costs
        throughput and is *not* needed for the seed study -- we want the genuine
        run-to-run spread of the method, not a suppressed one -- so it defaults to
        False. Turn it on only when chasing a reproducibility bug.

    Returns
    -------
    The seed, so call sites can record it inline.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pure-numpy call sites (e.g. the FEM-only Darcy adjoint)
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
