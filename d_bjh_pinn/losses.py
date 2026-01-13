"""Loss definitions for data fitting and physics constraints."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class LossWeights:
    """Weighting for loss components."""

    data: float = 1.0
    physics: float = 1.0
    boundary: float = 0.1
    smoothness: float = 0.1


def data_loss(predicted: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """Mean squared error for data points."""

    return tf.reduce_mean(tf.square(predicted - target))


def physics_residual_loss(residual: tf.Tensor) -> tf.Tensor:
    """Mean squared error for physics residual."""

    return tf.reduce_mean(tf.square(residual))


def boundary_condition_loss(values: tf.Tensor) -> tf.Tensor:
    """Placeholder for boundary condition loss."""

    return tf.reduce_mean(tf.square(values))


def smoothness_loss(values: tf.Tensor) -> tf.Tensor:
    """Placeholder for smoothness regularization."""

    return tf.reduce_mean(tf.square(values))


def total_loss(
    data_term: tf.Tensor,
    physics_term: tf.Tensor,
    boundary_term: tf.Tensor,
    smoothness_term: tf.Tensor,
    weights: LossWeights,
) -> tf.Tensor:
    """Combine weighted loss terms."""

    return (
        weights.data * data_term
        + weights.physics * physics_term
        + weights.boundary * boundary_term
        + weights.smoothness * smoothness_term
    )
