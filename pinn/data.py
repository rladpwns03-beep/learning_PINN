"""Data utilities for sampling collocation points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DomainBounds:
    """Lower/upper bounds for each dimension."""

    lower: Sequence[float]
    upper: Sequence[float]

    def to_tensor(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower/upper bounds as tensors on the given device."""

        lower_tensor = torch.tensor(self.lower, device=device, dtype=torch.float32)
        upper_tensor = torch.tensor(self.upper, device=device, dtype=torch.float32)
        return lower_tensor, upper_tensor


def sample_collocation_points(
    num_points: int,
    bounds: DomainBounds,
    device: torch.device,
) -> torch.Tensor:
    """Sample uniformly distributed collocation points within bounds."""

    lower, upper = bounds.to_tensor(device)
    dims = lower.numel()
    points = torch.rand((num_points, dims), device=device, dtype=torch.float32)
    return lower + (upper - lower) * points
