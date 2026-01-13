"""Differentiable Volterra operator builder."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VolterraConfig:
    """Configuration for the Volterra operator."""

    kernel_scale: float = 1.0


def build_volterra_operator(config: VolterraConfig) -> callable:
    """Return a callable that applies a Volterra integral operator.

    TODO: Replace with a proper discretized Volterra operator.
    """

    def operator(kernel_input: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        _ = kernel_input
        return config.kernel_scale * signal

    return operator
