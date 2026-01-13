"""Differentiable Volterra operator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

KernelCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

@dataclass(frozen=True)
class VolterraConfig:
    """Configuration for the Volterra operator."""

    kernel_scale: float = 1.0


def build_volterra_operator(config: VolterraConfig) -> KernelCallable:
    """Return a placeholder kernel K(p, r) for experimentation."""

    def kernel(pressure: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return config.kernel_scale * (pressure[:, None] + radius[None, :])

    return kernel


def _validate_inputs(
    pressure: torch.Tensor,
    r_grid: torch.Tensor,
    l_values: torch.Tensor,
    dr: torch.Tensor,
    j_idx: torch.Tensor,
    kernel_matrix: torch.Tensor,
) -> None:
    if pressure.ndim != 1:
        raise ValueError("pressure must be 1D (N,).")
    if r_grid.ndim != 1:
        raise ValueError("r_grid must be 1D (R,).")
    if l_values.ndim != 1:
        raise ValueError("l_values must be 1D (R,).")
    if dr.ndim != 1:
        raise ValueError("dr must be 1D (R,).")
    if j_idx.ndim != 1:
        raise ValueError("j_idx must be 1D (N,).")
    if pressure.shape[0] != j_idx.shape[0]:
        raise ValueError("pressure and j_idx must have the same length.")
    if r_grid.shape[0] != l_values.shape[0] or r_grid.shape[0] != dr.shape[0]:
        raise ValueError("r_grid, l_values, and dr must have length R.")
    if kernel_matrix.shape != (pressure.shape[0], r_grid.shape[0]):
        raise ValueError("kernel_matrix must be shape (N, R).")
    if j_idx.min().item() < 0 or j_idx.max().item() >= r_grid.shape[0]:
        raise ValueError("j_idx entries must be within [0, R-1].")


def _resolve_kernel(
    pressure: torch.Tensor,
    r_grid: torch.Tensor,
    kernel: KernelCallable | torch.Tensor,
) -> torch.Tensor:
    if callable(kernel):
        kernel_matrix = kernel(pressure, r_grid)
    else:
        kernel_matrix = kernel
    return kernel_matrix


def masked_matmul_volterra(
    pressure: torch.Tensor,
    r_grid: torch.Tensor,
    l_values: torch.Tensor,
    dr: torch.Tensor,
    kernel: KernelCallable | torch.Tensor,
    j_idx: torch.Tensor,
    *,
    local_a: torch.Tensor | None = None,
    local_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Volterra integral using a masked matmul approach.

    I(p) = sum_{k=j(p)}^{R-1} K(p, r_k) * L(r_k) * dr_k
    local(p) = A(p) * L(r_{local_idx})
    """

    kernel_matrix = _resolve_kernel(pressure, r_grid, kernel)
    _validate_inputs(pressure, r_grid, l_values, dr, j_idx, kernel_matrix)

    mask = torch.arange(r_grid.shape[0], device=r_grid.device)[None, :] >= j_idx[:, None]
    weights = kernel_matrix * l_values[None, :] * dr[None, :]
    integral = (weights * mask).sum(dim=1)

    if local_a is None:
        return integral

    idx = local_idx if local_idx is not None else j_idx
    local_term = local_a * l_values[idx]
    return integral + local_term


def cumsum_volterra(
    pressure: torch.Tensor,
    r_grid: torch.Tensor,
    l_values: torch.Tensor,
    dr: torch.Tensor,
    kernel: KernelCallable | torch.Tensor,
    j_idx: torch.Tensor,
    *,
    local_a: torch.Tensor | None = None,
    local_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Volterra integral using a reverse cumulative sum."""

    kernel_matrix = _resolve_kernel(pressure, r_grid, kernel)
    _validate_inputs(pressure, r_grid, l_values, dr, j_idx, kernel_matrix)

    integrand = kernel_matrix * l_values[None, :] * dr[None, :]
    rev = torch.flip(integrand, dims=[1])
    rev_cumsum = torch.cumsum(rev, dim=1)
    forward_cumsum = torch.flip(rev_cumsum, dims=[1])
    integral = forward_cumsum.gather(1, j_idx[:, None]).squeeze(1)

    if local_a is None:
        return integral

    idx = local_idx if local_idx is not None else j_idx
    local_term = local_a * l_values[idx]
    return integral + local_term
