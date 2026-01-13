"""Training loop for the D-BJH PINN project."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Adam

from d_bjh_pinn.config import DataConfig, ModelConfig, TrainConfig
from d_bjh_pinn.dataio import DatasetBundle, load_csv, make_batches
from d_bjh_pinn.losses import LossWeights, data_loss, physics_residual_loss, smoothness_loss, total_loss
from d_bjh_pinn.models.networks import LNet, NetworkConfig, ThetaNet
from d_bjh_pinn.physics.mapping import r_p
from d_bjh_pinn.physics.tcurve import ThicknessConfig
from d_bjh_pinn.physics.volterra import VolterraConfig, build_volterra_operator
from d_bjh_pinn.utils import set_deterministic


@dataclass(frozen=True)
class TrainState:
    """Container for training state."""

    theta_net: ThetaNet
    l_net: LNet
    losses: list[float]


def build_models(model_config: ModelConfig, device: torch.device) -> tuple[ThetaNet, LNet]:
    """Construct networks for theta and L."""

    theta_net = ThetaNet(NetworkConfig(model_config.theta_hidden)).to(device)
    l_net = LNet(NetworkConfig(model_config.l_hidden)).to(device)
    return theta_net, l_net


def build_physics_residual(
    pressure: torch.Tensor,
    l_values: torch.Tensor,
) -> torch.Tensor:
    """Compute physics residuals using Volterra operator.

    TODO: Implement real residual with Kelvin mapping and Volterra operator.
    """

    _ = pressure
    return l_values * 0.0


def train(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    loss_weights: LossWeights | None = None,
) -> TrainState:
    """Train the D-BJH PINN end-to-end."""

    set_deterministic(train_config.seed)
    device = train_config.torch_device()
    dataset = load_csv(data_config, device)
    theta_net, l_net = build_models(model_config, device)

    optimizer = Adam(
        list(theta_net.parameters()) + list(l_net.parameters()),
        lr=train_config.learning_rate,
    )
    loss_weights = loss_weights or LossWeights()

    losses: list[float] = []
    batches = make_batches([dataset.pressure, dataset.theta], train_config.batch_size)
    thickness = ThicknessConfig()
    volterra = build_volterra_operator(VolterraConfig())

    for epoch in range(train_config.epochs):
        for pressure_batch, theta_batch in batches:
            optimizer.zero_grad(set_to_none=True)
            theta_pred = theta_net(pressure_batch)
            radius = r_p(pressure_batch, thickness=thickness)
            l_pred = l_net(radius)

            _ = volterra(radius, l_pred)
            residual = build_physics_residual(pressure_batch, l_pred)

            data_term = data_loss(theta_pred, theta_batch)
            physics_term = physics_residual_loss(residual)
            smoothness_term = smoothness_loss(l_pred)
            loss = total_loss(
                data_term,
                physics_term,
                torch.tensor(0.0, device=device),
                smoothness_term,
                loss_weights,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())

        if (epoch + 1) % train_config.log_every == 0:
            print(f"epoch={epoch + 1} loss={losses[-1]:.6f}")

    return TrainState(theta_net=theta_net, l_net=l_net, losses=losses)


def save_checkpoint(state: TrainState, path: str) -> None:
    """Save model checkpoints to disk."""

    torch.save(
        {
            "theta_state": state.theta_net.state_dict(),
            "l_state": state.l_net.state_dict(),
            "losses": state.losses,
        },
        path,
    )
