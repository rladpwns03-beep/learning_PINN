import torch

from d_bjh_pinn.physics.volterra import (
    VolterraConfig,
    build_volterra_operator,
    cumsum_volterra,
    masked_matmul_volterra,
)


def _naive_volterra(
    pressure: torch.Tensor,
    r_grid: torch.Tensor,
    l_values: torch.Tensor,
    dr: torch.Tensor,
    kernel: torch.Tensor,
    j_idx: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for i in range(pressure.shape[0]):
        total = torch.tensor(0.0, dtype=pressure.dtype, device=pressure.device)
        for k in range(j_idx[i].item(), r_grid.shape[0]):
            total = total + kernel[i, k] * l_values[k] * dr[k]
        outputs.append(total)
    return torch.stack(outputs)


def test_volterra_matches_naive() -> None:
    pressure = torch.tensor([0.1, 0.3], dtype=torch.float32)
    r_grid = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    l_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    dr = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    j_idx = torch.tensor([0, 1], dtype=torch.long)

    kernel = build_volterra_operator(VolterraConfig(kernel_scale=2.0))(pressure, r_grid)
    expected = _naive_volterra(pressure, r_grid, l_values, dr, kernel, j_idx)

    masked = masked_matmul_volterra(pressure, r_grid, l_values, dr, kernel, j_idx)
    cumsum = cumsum_volterra(pressure, r_grid, l_values, dr, kernel, j_idx)

    assert torch.allclose(masked, expected)
    assert torch.allclose(cumsum, expected)


def test_masked_matmul_gradcheck() -> None:
    pressure = torch.tensor([0.2, 0.5], dtype=torch.float64)
    r_grid = torch.tensor([0.1, 0.3, 0.7], dtype=torch.float64)
    dr = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    j_idx = torch.tensor([0, 1], dtype=torch.long)

    def kernel(p: torch.Tensor, r: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return scale * (p[:, None] + r[None, :])

    def func(l_values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return masked_matmul_volterra(
            pressure,
            r_grid,
            l_values,
            dr,
            lambda p, r: kernel(p, r, scale),
            j_idx,
        )

    l_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func, (l_values, scale))
