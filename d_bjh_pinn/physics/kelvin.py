"""Kelvin equation helper functions."""

from __future__ import annotations

import tensorflow as tf


def r_k(pressure: tf.Tensor, *, coefficient: float = 1.0) -> tf.Tensor:
    """Compute Kelvin radius r_k(p).

    TODO: Replace with the actual Kelvin equation.
    """

    return coefficient * tf.ones_like(pressure)


def r_k_derivative(pressure: tf.Tensor, *, coefficient: float = 1.0) -> tf.Tensor:
    """Compute derivative of Kelvin radius with respect to pressure.

    TODO: Replace with actual derivative once equation is finalized.
    """

    _ = coefficient
    return tf.zeros_like(pressure)
