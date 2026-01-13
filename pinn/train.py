"""Training loop for the PINN model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.optim import Adam

from pinn.config import TrainConfig
from pinn.data import DomainBounds, sample_collocation_points
from pinn.physics import DBJHParameters, compute_dbjh_residual
from pinn.utils import set_deterministic

ResidualFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainState:
    """Container for training outputs."""

    model: nn.Module
    losses: list[float]


def build_residual_fn(parameters: DBJHParameters) -> ResidualFn:
    """Build the residual function for training."""

    def residual_fn(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return compute_dbjh_residual(inputs, outputs, parameters)

    return residual_fn


def train_model(config: TrainConfig, *, parameters: DBJHParameters | None = None) -> TrainState:
    """Train a PINN model end-to-end.

    Args:
        config: Training configuration.
        parameters: Optional D-BJH parameter bundle.
    """

    set_deterministic(config.seed)
    device = config.torch_device()
    parameters = parameters or DBJHParameters()

    input_dim = len(config.domain_lower)
    model = nn.Sequential()
    from pinn.model import build_model  # local import to avoid circular

    model = build_model(input_dim, config.hidden_layers, output_dim=1)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    residual_fn = build_residual_fn(parameters)
    bounds = DomainBounds(config.domain_lower, config.domain_upper)

    losses: list[float] = []
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        collocation = sample_collocation_points(config.num_collocation, bounds, device)
        collocation.requires_grad_(True)
        outputs = model(collocation)
        residual = residual_fn(collocation, outputs)
        loss = torch.mean(residual**2)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())

    return TrainState(model=model, losses=losses)
