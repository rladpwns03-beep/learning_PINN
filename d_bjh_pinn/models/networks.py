"""Neural network definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for shared MLP structure."""

    hidden_layers: Sequence[int]


class ThetaNet(nn.Module):
    """Predicts theta values from pressure inputs."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.network = _build_mlp(input_dim=1, output_dim=1, hidden_layers=config.hidden_layers)

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        return self.network(pressure)


class LNet(nn.Module):
    """Predicts non-negative L values from radius inputs."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.network = _build_mlp(input_dim=1, output_dim=1, hidden_layers=config.hidden_layers)
        self.softplus = nn.Softplus()

    def forward(self, radius: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.network(radius))


def _build_mlp(input_dim: int, output_dim: int, hidden_layers: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.Tanh())
        in_features = width
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)
