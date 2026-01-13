"""Utility helpers for reproducibility and logging."""

from __future__ import annotations

import random
from typing import Optional

import torch


def set_deterministic(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Seed random generators and configure deterministic torch behavior.

    Args:
        seed: Random seed for Python and torch.
        deterministic_algorithms: If True, enable deterministic torch ops.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device: Optional[str]) -> torch.device:
    """Resolve a torch.device from an optional string."""

    if device is None:
        return torch.device("cpu")
    return torch.device(device)
