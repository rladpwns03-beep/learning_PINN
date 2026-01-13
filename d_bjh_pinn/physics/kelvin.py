"""Kelvin equation helper functions."""

from __future__ import annotations

import torch


def r_k(pressure: torch.Tensor, *, coefficient: float = 1.0) -> torch.Tensor:
    """Compute Kelvin radius r_k(p).

    TODO: Replace with the actual Kelvin equation.
    """

    return coefficient * torch.ones_like(pressure)


def r_k_derivative(pressure: torch.Tensor, *, coefficient: float = 1.0) -> torch.Tensor:
    """Compute derivative of Kelvin radius with respect to pressure.

    TODO: Replace with actual derivative once equation is finalized.
    """

    _ = coefficient
    return torch.zeros_like(pressure)
