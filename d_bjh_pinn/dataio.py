"""CSV loading and preprocessing utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Sequence

import torch

from d_bjh_pinn.config import DataConfig


@dataclass(frozen=True)
class DatasetBundle:
    """Container for loaded tensors and normalization metadata."""

    pressure: torch.Tensor
    theta: torch.Tensor
    pressure_mean: torch.Tensor
    pressure_std: torch.Tensor
    theta_mean: torch.Tensor
    theta_std: torch.Tensor


def _normalize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (tensor - mean) / std, mean, std


def load_csv(config: DataConfig, device: torch.device) -> DatasetBundle:
    """Load a CSV file into tensors with optional normalization."""

    pressures: list[float] = []
    thetas: list[float] = []
    with open(config.csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pressures.append(float(row[config.p_rel_column]))
            thetas.append(float(row[config.theta_column]))

    pressure_tensor = torch.tensor(pressures, dtype=torch.float32, device=device).unsqueeze(-1)
    theta_tensor = torch.tensor(thetas, dtype=torch.float32, device=device).unsqueeze(-1)

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

    zeros = torch.zeros((1, 1), dtype=torch.float32, device=device)
    ones = torch.ones((1, 1), dtype=torch.float32, device=device)
    return DatasetBundle(
        pressure=pressure_tensor,
        theta=theta_tensor,
        pressure_mean=zeros,
        pressure_std=ones,
        theta_mean=zeros,
        theta_std=ones,
    )


def make_batches(
    tensors: Sequence[torch.Tensor],
    batch_size: int,
) -> list[tuple[torch.Tensor, ...]]:
    """Create simple batches from tensors using slicing."""

    if not tensors:
        return []
    num_samples = tensors[0].shape[0]
    batches: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch = tuple(tensor[start:end] for tensor in tensors)
        batches.append(batch)
    return batches
