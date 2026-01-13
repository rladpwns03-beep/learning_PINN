"""Loss definitions for data fitting and physics constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossWeights:
    """Weighting for loss components."""

    data: float = 1.0
    physics: float = 1.0
    boundary: float = 0.1
    smoothness: float = 0.1


def data_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error for data points."""

    return torch.mean((predicted - target) ** 2)


def physics_residual_loss(residual: torch.Tensor) -> torch.Tensor:
    """Mean squared error for physics residual."""

    return torch.mean(residual**2)


def boundary_condition_loss(values: torch.Tensor) -> torch.Tensor:
    """Placeholder for boundary condition loss."""

    return torch.mean(values**2)


def smoothness_loss(values: torch.Tensor) -> torch.Tensor:
    """Placeholder for smoothness regularization."""

    return torch.mean(values**2)


def total_loss(
    data_term: torch.Tensor,
    physics_term: torch.Tensor,
    boundary_term: torch.Tensor,
    smoothness_term: torch.Tensor,
    weights: LossWeights,
) -> torch.Tensor:
    """Combine weighted loss terms."""

    return (
        weights.data * data_term
        + weights.physics * physics_term
        + weights.boundary * boundary_term
        + weights.smoothness * smoothness_term
    )
