"""Evaluation utilities for PSD metrics and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Metrics:
    """Placeholder metrics container."""

    specific_surface_area: torch.Tensor
    pore_volume: torch.Tensor
    mean_diameter: torch.Tensor


def compute_metrics(l_values: torch.Tensor) -> Metrics:
    """Compute PSD-derived metrics.

    TODO: Replace with real SSA/PV/MD calculations.
    """

    zeros = torch.zeros((1,), device=l_values.device, dtype=l_values.dtype)
    return Metrics(
        specific_surface_area=zeros,
        pore_volume=zeros,
        mean_diameter=zeros,
    )


def reconstruct_psd(l_values: torch.Tensor) -> torch.Tensor:
    """Reconstruct PSD from L values.

    TODO: Replace with reconstruction logic.
    """

    return l_values
