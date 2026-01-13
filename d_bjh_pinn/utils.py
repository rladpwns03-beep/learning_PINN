"""Utility helpers for deterministic behavior."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Set deterministic behavior for Python and torch."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
