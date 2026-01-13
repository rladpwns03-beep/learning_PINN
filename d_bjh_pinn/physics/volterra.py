"""Differentiable Volterra operator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

KernelCallable = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

@dataclass(frozen=True)
class VolterraConfig:
    """Configuration for the Volterra operator."""

    kernel_scale: float = 1.0


def build_volterra_operator(config: VolterraConfig) -> KernelCallable:
    """Return a placeholder kernel K(p, r) for experimentation."""

    def kernel(pressure: tf.Tensor, radius: tf.Tensor) -> tf.Tensor:
        return config.kernel_scale * (pressure[:, None] + radius[None, :])

    return kernel


def _validate_inputs(
    pressure: tf.Tensor,
    r_grid: tf.Tensor,
    l_values: tf.Tensor,
    dr: tf.Tensor,
    j_idx: tf.Tensor,
    kernel_matrix: tf.Tensor,
) -> None:
    if pressure.shape.rank != 1:
        raise ValueError("pressure must be 1D (N,).")
    if r_grid.shape.rank != 1:
        raise ValueError("r_grid must be 1D (R,).")
    if l_values.shape.rank != 1:
        raise ValueError("l_values must be 1D (R,).")
    if dr.shape.rank != 1:
        raise ValueError("dr must be 1D (R,).")
    if j_idx.shape.rank != 1:
        raise ValueError("j_idx must be 1D (N,).")
    if pressure.shape[0] != j_idx.shape[0]:
        raise ValueError("pressure and j_idx must have the same length.")
    if r_grid.shape[0] != l_values.shape[0] or r_grid.shape[0] != dr.shape[0]:
        raise ValueError("r_grid, l_values, and dr must have length R.")
    if kernel_matrix.shape != (pressure.shape[0], r_grid.shape[0]):
        raise ValueError("kernel_matrix must be shape (N, R).")
    j_min = int(tf.reduce_min(j_idx).numpy())
    j_max = int(tf.reduce_max(j_idx).numpy())
    if j_min < 0 or j_max >= int(r_grid.shape[0]):
        raise ValueError("j_idx entries must be within [0, R-1].")


def _resolve_kernel(
    pressure: tf.Tensor,
    r_grid: tf.Tensor,
    kernel: KernelCallable | tf.Tensor,
) -> tf.Tensor:
    if callable(kernel):
        kernel_matrix = kernel(pressure, r_grid)
    else:
        kernel_matrix = kernel
    return kernel_matrix


def masked_matmul_volterra(
    pressure: tf.Tensor,
    r_grid: tf.Tensor,
    l_values: tf.Tensor,
    dr: tf.Tensor,
    kernel: KernelCallable | tf.Tensor,
    j_idx: tf.Tensor,
    *,
    local_a: tf.Tensor | None = None,
    local_idx: tf.Tensor | None = None,
) -> tf.Tensor:
    """Compute Volterra integral using a masked matmul approach.

    I(p) = sum_{k=j(p)}^{R-1} K(p, r_k) * L(r_k) * dr_k
    local(p) = A(p) * L(r_{local_idx})
    """

    kernel_matrix = _resolve_kernel(pressure, r_grid, kernel)
    _validate_inputs(pressure, r_grid, l_values, dr, j_idx, kernel_matrix)

    mask = tf.range(r_grid.shape[0])[None, :] >= j_idx[:, None]
    weights = kernel_matrix * l_values[None, :] * dr[None, :]
    integral = tf.reduce_sum(weights * tf.cast(mask, weights.dtype), axis=1)

    if local_a is None:
        return integral

    idx = local_idx if local_idx is not None else j_idx
    local_term = local_a * tf.gather(l_values, idx)
    return integral + local_term


def cumsum_volterra(
    pressure: tf.Tensor,
    r_grid: tf.Tensor,
    l_values: tf.Tensor,
    dr: tf.Tensor,
    kernel: KernelCallable | tf.Tensor,
    j_idx: tf.Tensor,
    *,
    local_a: tf.Tensor | None = None,
    local_idx: tf.Tensor | None = None,
) -> tf.Tensor:
    """Compute Volterra integral using a reverse cumulative sum."""

    kernel_matrix = _resolve_kernel(pressure, r_grid, kernel)
    _validate_inputs(pressure, r_grid, l_values, dr, j_idx, kernel_matrix)

    integrand = kernel_matrix * l_values[None, :] * dr[None, :]
    rev = tf.reverse(integrand, axis=[1])
    rev_cumsum = tf.cumsum(rev, axis=1)
    forward_cumsum = tf.reverse(rev_cumsum, axis=[1])
    integral = tf.gather(forward_cumsum, j_idx[:, None], axis=1, batch_dims=1)
    integral = tf.squeeze(integral, axis=1)

    if local_a is None:
        return integral

    idx = local_idx if local_idx is not None else j_idx
    local_term = local_a * tf.gather(l_values, idx)
    return integral + local_term
