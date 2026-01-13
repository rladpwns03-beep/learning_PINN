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
    normalize_input: bool = True
    enforce_positive_output: bool = True
    output_log: bool = False
    input_min: float = 0.0
    input_max: float = 1.0


class ThetaNet(nn.Module):
    """Predicts theta values from relative pressure inputs."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _build_mlp(input_dim=1, output_dim=1, hidden_layers=config.hidden_layers)
        self.softplus = nn.Softplus()
        _init_weights(self.network)

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        inputs = _normalize_input(pressure, self.config)
        outputs = self.network(inputs)
        if self.config.enforce_positive_output:
            outputs = self.softplus(outputs)
        return outputs


class LNet(nn.Module):
    """Predicts non-negative L values from radius inputs."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _build_mlp(input_dim=1, output_dim=1, hidden_layers=config.hidden_layers)
        self.softplus = nn.Softplus()
        _init_weights(self.network)

    def forward(self, radius: torch.Tensor) -> torch.Tensor:
        inputs = _normalize_input(radius, self.config)
        outputs = self.network(inputs)
        if self.config.output_log:
            outputs = torch.exp(outputs)
        else:
            outputs = self.softplus(outputs)
        return outputs


def dtheta_dp(theta_net: ThetaNet, pressure: torch.Tensor) -> torch.Tensor:
    """Compute dTheta/dp via autograd."""

    pressure = pressure.clone().detach().requires_grad_(True)
    theta = theta_net(pressure)
    grads = torch.autograd.grad(
        outputs=theta,
        inputs=pressure,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
    )[0]
    return grads


def _build_mlp(input_dim: int, output_dim: int, hidden_layers: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.Tanh())
        in_features = width
    layers.append(nn.Linear(in_features, output_dim))
    return nn.Sequential(*layers)


def _normalize_input(inputs: torch.Tensor, config: NetworkConfig) -> torch.Tensor:
    if not config.normalize_input:
        return inputs
    denom = max(config.input_max - config.input_min, 1e-8)
    return (inputs - config.input_min) / denom


def _init_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
