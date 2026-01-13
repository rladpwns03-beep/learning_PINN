"""Mapping utilities for Kelvin radius and thickness curves."""

from __future__ import annotations

import tensorflow as tf

from d_bjh_pinn.physics.kelvin import r_k
from d_bjh_pinn.physics.tcurve import ThicknessConfig, t_curve


def r_p(pressure: tf.Tensor, *, thickness: ThicknessConfig, coefficient: float = 1.0) -> tf.Tensor:
    """Compute pore radius r_p(p) = r_k(p) + t(p)."""

    return r_k(pressure, coefficient=coefficient) + t_curve(pressure, thickness)
