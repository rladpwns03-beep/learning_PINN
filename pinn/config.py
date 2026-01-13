"""Configuration dataclasses for the PINN pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training a PINN model.

    Attributes:
        seed: RNG seed for reproducible runs.
        device: Torch device specifier (e.g., "cpu", "cuda", "cuda:0").
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        hidden_layers: Hidden layer widths for the MLP.
        num_collocation: Number of collocation points sampled per epoch.
        domain_lower: Lower bounds for the domain (1D/2D/etc).
        domain_upper: Upper bounds for the domain (same length as domain_lower).
    """

    seed: int = 7
    device: str = "cpu"
    epochs: int = 1000
    learning_rate: float = 1e-3
    hidden_layers: Sequence[int] = (64, 64, 64)
    num_collocation: int = 512
    domain_lower: Sequence[float] = (0.0, 0.0)
    domain_upper: Sequence[float] = (1.0, 1.0)

    def torch_device(self) -> torch.device:
        """Return the torch device for this configuration."""

        return torch.device(self.device)
