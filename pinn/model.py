"""Neural network model definitions for the PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the PINN MLP."""

    input_dim: int
    output_dim: int
    hidden_layers: Sequence[int]


class PinnModel(nn.Module):
    """Multi-layer perceptron used as the PINN surrogate."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        in_features = config.input_dim
        for width in config.hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.Tanh())
            in_features = width
        layers.append(nn.Linear(in_features, config.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            inputs: Tensor of shape (batch, input_dim).
        """

        return self.network(inputs)


def build_model(input_dim: int, hidden_layers: Iterable[int], output_dim: int) -> PinnModel:
    """Convenience function to build a PINN model."""

    config = ModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=tuple(hidden_layers),
    )
    return PinnModel(config)
