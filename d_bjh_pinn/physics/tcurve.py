"""Thickness curve definition for t(p)."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ThicknessConfig:
    """Configuration for t(p) thickness curve."""

    coefficient: float = 1.0


def t_curve(pressure: tf.Tensor, config: ThicknessConfig) -> tf.Tensor:
    """Compute thickness curve t(p).

    TODO: Replace with Kruk or other equation once specified.
    """

    return config.coefficient * tf.ones_like(pressure)
