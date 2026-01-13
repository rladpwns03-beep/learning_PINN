import torch

from pinn.physics import DBJHParameters, compute_dbjh_residual


def test_dbjh_residual_shape() -> None:
    inputs = torch.zeros((5, 2), requires_grad=True)
    outputs = torch.ones((5, 1))
    residual = compute_dbjh_residual(inputs, outputs, DBJHParameters())
    assert residual.shape == outputs.shape
