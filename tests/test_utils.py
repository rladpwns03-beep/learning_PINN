import torch

from pinn.model import build_model
from pinn.utils import set_deterministic


def test_set_deterministic_reproducible() -> None:
    set_deterministic(123)
    first = torch.rand(3)
    set_deterministic(123)
    second = torch.rand(3)
    assert torch.allclose(first, second)


def test_model_forward_shape() -> None:
    model = build_model(input_dim=2, hidden_layers=[8, 8], output_dim=1)
    inputs = torch.zeros((4, 2))
    outputs = model(inputs)
    assert outputs.shape == (4, 1)
