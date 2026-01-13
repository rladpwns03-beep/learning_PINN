"""Configuration dataclasses for the D-BJH PINN project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing configuration."""

    csv_path: str
    p_rel_column: str = "p/p0"
    theta_column: str = "Va/cm3(STP)g-1"
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

    def device_name(self) -> str:
        """Return the TensorFlow device name for this configuration."""

        return self.device
