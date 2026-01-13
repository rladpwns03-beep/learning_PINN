"""Physics residual definitions for the D-BJH-based PINN."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DBJHParameters:
    """Placeholder parameters for the D-BJH equation.

    TODO: Replace with real D-BJH equation parameters once finalized.
    """

    coefficient_a: float = 1.0
    coefficient_b: float = 1.0


def compute_dbjh_residual(
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    parameters: DBJHParameters,
) -> torch.Tensor:
    """Compute the residual of the D-BJH equation.

    Args:
        inputs: Input coordinates tensor of shape (batch, dim).
        predictions: Model outputs of shape (batch, 1) or (batch, output_dim).
        parameters: Parameter bundle for the D-BJH equation.

    Returns:
        Tensor of residuals with the same shape as predictions.

    TODO: Implement the actual D-BJH residual using torch autograd ops only.
    """

    _ = parameters
    return torch.zeros_like(predictions)
