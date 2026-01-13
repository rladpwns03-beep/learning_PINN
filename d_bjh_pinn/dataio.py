"""CSV loading and preprocessing utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf

from d_bjh_pinn.config import DataConfig


@dataclass(frozen=True)
class DatasetBundle:
    """Container for loaded tensors and normalization metadata."""

    pressure: tf.Tensor
    theta: tf.Tensor
    pressure_mean: tf.Tensor
    pressure_std: tf.Tensor
    theta_mean: tf.Tensor
    theta_std: tf.Tensor


def _normalize(tensor: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(tensor, axis=0, keepdims=True)
    std = tf.math.reduce_std(tensor, axis=0, keepdims=True)
    std = tf.maximum(std, tf.constant(1e-8, dtype=tensor.dtype))
    return (tensor - mean) / std, mean, std


def load_csv(config: DataConfig, device: str) -> DatasetBundle:
    """Load a CSV file into tensors with optional normalization."""

    pressures: list[float] = []
    thetas: list[float] = []
    with open(config.csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pressures.append(float(row[config.p_rel_column]))
            thetas.append(float(row[config.theta_column]))

    with tf.device(device):
        pressure_tensor = tf.convert_to_tensor(pressures, dtype=tf.float32)
        theta_tensor = tf.convert_to_tensor(thetas, dtype=tf.float32)
        pressure_tensor = tf.expand_dims(pressure_tensor, axis=-1)
        theta_tensor = tf.expand_dims(theta_tensor, axis=-1)

    if config.normalize:
        pressure_norm, pressure_mean, pressure_std = _normalize(pressure_tensor)
        theta_norm, theta_mean, theta_std = _normalize(theta_tensor)
        return DatasetBundle(
            pressure=pressure_norm,
            theta=theta_norm,
            pressure_mean=pressure_mean,
            pressure_std=pressure_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
        )

    zeros = tf.zeros((1, 1), dtype=tf.float32)
    ones = tf.ones((1, 1), dtype=tf.float32)
    return DatasetBundle(
        pressure=pressure_tensor,
        theta=theta_tensor,
        pressure_mean=zeros,
        pressure_std=ones,
        theta_mean=zeros,
        theta_std=ones,
    )


def make_batches(
    tensors: Sequence[tf.Tensor],
    batch_size: int,
) -> list[tuple[tf.Tensor, ...]]:
    """Create simple batches from tensors using slicing."""

    if not tensors:
        return []
    num_samples = int(tensors[0].shape[0])
    batches: list[tuple[tf.Tensor, ...]] = []
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch = tuple(tensor[start:end] for tensor in tensors)
        batches.append(batch)
    return batches
