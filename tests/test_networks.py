import torch

from d_bjh_pinn.models.networks import LNet, NetworkConfig, ThetaNet, dtheta_dp


def test_theta_net_output_shape() -> None:
    model = ThetaNet(NetworkConfig(hidden_layers=(8, 8)))
    inputs = torch.zeros((5, 1))
    outputs = model(inputs)
    assert outputs.shape == (5, 1)


def test_l_net_non_negative() -> None:
    model = LNet(NetworkConfig(hidden_layers=(8, 8)))
    inputs = torch.zeros((5, 1))
    outputs = model(inputs)
    assert torch.all(outputs >= 0)


def test_dtheta_dp_shape_and_finite() -> None:
    model = ThetaNet(NetworkConfig(hidden_layers=(8, 8)))
    inputs = torch.rand((4, 1))
    grads = dtheta_dp(model, inputs)
    assert grads.shape == inputs.shape
    assert torch.isfinite(grads).all()
