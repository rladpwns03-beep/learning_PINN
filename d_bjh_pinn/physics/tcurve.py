"""Thickness curve definition for t(p)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThicknessConfig:
    """Configuration for t(p) thickness curve."""

    coefficient: float = 1.0


def t_curve(pressure: torch.Tensor, config: ThicknessConfig) -> torch.Tensor:
    """Compute thickness curve t(p).

    TODO: Replace with Kruk or other equation once specified.
    """

    return config.coefficient * torch.ones_like(pressure)
