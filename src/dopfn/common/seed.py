"""Seeding for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy and PyTorch (CPU and CUDA).

    Prefer explicit ``seed`` / ``torch.Generator`` arguments where a function offers them; this
    only fixes the global RNG state.

    Args:
        seed: Global seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
