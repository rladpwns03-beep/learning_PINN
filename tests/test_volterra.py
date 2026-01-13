import torch

from d_bjh_pinn.physics.volterra import VolterraConfig, build_volterra_operator


def test_volterra_operator_identity() -> None:
    operator = build_volterra_operator(VolterraConfig(kernel_scale=2.0))
    kernel_input = torch.zeros((3, 1))
    signal = torch.ones((3, 1))
    output = operator(kernel_input, signal)
    assert torch.allclose(output, signal * 2.0)
