"""Top-level package for the D-BJH-based PINN project."""

from pinn.config import TrainConfig
from pinn.model import PinnModel

__all__ = ["PinnModel", "TrainConfig"]
