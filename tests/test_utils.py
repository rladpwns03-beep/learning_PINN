import tensorflow as tf

from d_bjh_pinn.models.networks import NetworkConfig, ThetaNet
from d_bjh_pinn.utils import seed_everything


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    first = tf.random.uniform((3,))
    seed_everything(123)
    second = tf.random.uniform((3,))
    tf.debugging.assert_near(first, second)


def test_model_forward_shape() -> None:
    model = ThetaNet(NetworkConfig(hidden_layers=(8, 8)))
    inputs = tf.zeros((4, 1))
    outputs = model(inputs)
    assert outputs.shape == (4, 1)
