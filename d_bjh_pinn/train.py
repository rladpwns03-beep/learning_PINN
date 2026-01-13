"""Training loop for the D-BJH PINN project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import Adam

from d_bjh_pinn.config import DataConfig, ModelConfig, TrainConfig
from d_bjh_pinn.dataio import DatasetBundle, load_csv, make_batches
from d_bjh_pinn.losses import LossWeights, data_loss, physics_residual_loss, smoothness_loss, total_loss
from d_bjh_pinn.models.networks import LNet, NetworkConfig, ThetaNet
from d_bjh_pinn.physics.mapping import r_p
from d_bjh_pinn.physics.tcurve import ThicknessConfig
from d_bjh_pinn.physics.volterra import VolterraConfig, build_volterra_operator, masked_matmul_volterra
from d_bjh_pinn.utils import seed_everything


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
    r_grid: torch.Tensor,
    l_values: torch.Tensor,
    dr: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute physics residuals using Volterra operator.

    TODO: Implement real residual with Kelvin mapping and Volterra operator.
    Expected inputs:
        pressure: (N,) collocation pressures.
        r_grid: (R,) radius grid.
        l_values: (R,) L values on the grid.
        dr: (R,) radius grid spacing.
        j_idx: (N,) lower limit indices into r_grid for each pressure.
    Returns:
        (N,) residual vector.
    """

    kernel = build_volterra_operator(VolterraConfig())
    integral = masked_matmul_volterra(pressure, r_grid, l_values, dr, kernel, j_idx)
    return integral * 0.0


def _make_collocation_points(num_points: int, device: torch.device) -> torch.Tensor:
    """Create log-spaced collocation points in (0, 1]."""

    p_min = 1e-6
    p_max = 1.0
    log_space = torch.linspace(torch.log10(torch.tensor(p_min)), 0.0, num_points, device=device)
    return torch.pow(10.0, log_space)


def _build_r_grid(r_min: float, r_max: float, num_points: int, device: torch.device) -> torch.Tensor:
    """Create a radius grid."""

    return torch.linspace(r_min, r_max, num_points, device=device)


def _map_p_to_j_idx(radius: torch.Tensor, r_grid: torch.Tensor) -> torch.Tensor:
    """Map radii to nearest indices in r_grid."""

    diff = torch.abs(radius[:, None] - r_grid[None, :])
    return torch.argmin(diff, dim=1)


def _compute_dr(r_grid: torch.Tensor) -> torch.Tensor:
    dr = torch.zeros_like(r_grid)
    dr[:-1] = r_grid[1:] - r_grid[:-1]
    dr[-1] = dr[-2]
    return dr


def _log_csv(path: Path, header: Iterable[str], values: Iterable[float], *, write_header: bool) -> None:
    line = ",".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in values)
    if write_header:
        path.write_text(",".join(header) + "\n" + line + "\n")
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def train(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    loss_weights: LossWeights | None = None,
    outdir: Path | None = None,
    use_lbfgs: bool = False,
) -> tuple[TrainState, dict[str, list[float]]]:
    """Train the D-BJH PINN end-to-end."""

    seed_everything(train_config.seed)
    device = train_config.torch_device()
    dataset = load_csv(data_config, device)
    theta_net, l_net = build_models(model_config, device)

    optimizer = Adam(
        list(theta_net.parameters()) + list(l_net.parameters()),
        lr=train_config.learning_rate,
    )
    loss_weights = loss_weights or LossWeights()

    outdir = outdir or Path("runs/default")
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_log.csv"
    best_path = outdir / "best.pt"

    losses: list[float] = []
    history: dict[str, list[float]] = {"total": [], "data": [], "physics": [], "smoothness": []}
    batches = make_batches([dataset.pressure, dataset.theta], train_config.batch_size)
    thickness = ThicknessConfig()

    radius_data = r_p(dataset.pressure, thickness=thickness).squeeze(1)
    r_min = radius_data.min().item()
    r_max = radius_data.max().item()
    r_grid = _build_r_grid(r_min=r_min, r_max=r_max, num_points=128, device=device)
    dr = _compute_dr(r_grid)

    best_loss = float("inf")
    wrote_header = False
    for epoch in range(train_config.epochs):
        for pressure_batch, theta_batch in batches:
            optimizer.zero_grad(set_to_none=True)
            theta_pred = theta_net(pressure_batch)

            radius = r_p(pressure_batch, thickness=thickness)
            l_pred = l_net(radius)

            collocation = _make_collocation_points(train_config.batch_size, device)
            radius_c = r_p(collocation, thickness=thickness).squeeze(1)
            j_idx = _map_p_to_j_idx(radius_c, r_grid)
            l_grid = l_net(r_grid.unsqueeze(1)).squeeze(1)
            residual = build_physics_residual(collocation, r_grid, l_grid, dr, j_idx)

            data_term = data_loss(theta_pred, theta_batch)
            physics_term = physics_residual_loss(residual)
            smoothness_term = smoothness_loss(l_pred)
            total = total_loss(
                data_term,
                physics_term,
                torch.tensor(0.0, device=device),
                smoothness_term,
                loss_weights,
            )
            total.backward()
            optimizer.step()

            total_value = total.detach().cpu().item()
            losses.append(total_value)
            history["total"].append(total_value)
            history["data"].append(data_term.detach().cpu().item())
            history["physics"].append(physics_term.detach().cpu().item())
            history["smoothness"].append(smoothness_term.detach().cpu().item())

        if total_value < best_loss:
            best_loss = total_value
            save_checkpoint(TrainState(theta_net=theta_net, l_net=l_net, losses=losses), best_path)

        if (epoch + 1) % train_config.log_every == 0:
            print(f"epoch={epoch + 1} loss={total_value:.6f}")
            _log_csv(
                log_path,
                ["epoch", "total", "data", "physics", "smoothness"],
                [epoch + 1, total_value, history["data"][-1], history["physics"][-1], history["smoothness"][-1]],
                write_header=not wrote_header,
            )
            wrote_header = True

    if use_lbfgs:
        lbfgs = torch.optim.LBFGS(list(theta_net.parameters()) + list(l_net.parameters()))

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            pressure_batch, theta_batch = batches[0]
            theta_pred = theta_net(pressure_batch)
            radius = r_p(pressure_batch, thickness=thickness)
            l_pred = l_net(radius)
            collocation = _make_collocation_points(train_config.batch_size, device)
            radius_c = r_p(collocation, thickness=thickness).squeeze(1)
            j_idx = _map_p_to_j_idx(radius_c, r_grid)
            l_grid = l_net(r_grid.unsqueeze(1)).squeeze(1)
            residual = build_physics_residual(collocation, r_grid, l_grid, dr, j_idx)
            data_term = data_loss(theta_pred, theta_batch)
            physics_term = physics_residual_loss(residual)
            smoothness_term = smoothness_loss(l_pred)
            total = total_loss(
                data_term,
                physics_term,
                torch.tensor(0.0, device=device),
                smoothness_term,
                loss_weights,
            )
            total.backward()
            return total

        lbfgs.step(closure)

    state = TrainState(theta_net=theta_net, l_net=l_net, losses=losses)
    return state, history


def save_checkpoint(state: TrainState, path: str | Path) -> None:
    """Save model checkpoints to disk."""

    torch.save(
        {
            "theta_state": state.theta_net.state_dict(),
            "l_state": state.l_net.state_dict(),
            "losses": state.losses,
        },
        str(path),
    )
