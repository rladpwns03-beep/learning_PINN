"""Configuration dataclasses for the D-BJH PINN project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing configuration."""

    csv_path: str
    pressure_column: str = "pressure"
    theta_column: str = "theta"
    normalize: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""

    theta_hidden: Sequence[int] = (64, 64)
    l_hidden: Sequence[int] = (64, 64)


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    seed: int = 7
    device: str = "cpu"
    epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 128
    log_every: int = 100

    def torch_device(self) -> torch.device:
        """Return the torch device for this configuration."""

        return torch.device(self.device)
