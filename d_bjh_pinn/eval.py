"""Evaluation utilities for PSD metrics and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class Metrics:
    """Placeholder metrics container."""

    specific_surface_area: tf.Tensor
    pore_volume: tf.Tensor
    mean_diameter: tf.Tensor


def compute_metrics(l_values: tf.Tensor) -> Metrics:
    """Compute PSD-derived metrics.

    TODO: Replace with real SSA/PV/MD calculations.
    """

    zeros = tf.zeros((1,), dtype=l_values.dtype)
    return Metrics(
        specific_surface_area=zeros,
        pore_volume=zeros,
        mean_diameter=zeros,
    )


def reconstruct_psd(l_values: tf.Tensor) -> tf.Tensor:
    """Reconstruct PSD from L values.

    TODO: Replace with reconstruction logic.
    """

    return l_values
