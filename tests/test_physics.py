import tensorflow as tf

from d_bjh_pinn.physics.kelvin import r_k


def test_kelvin_shape() -> None:
    inputs = tf.zeros((5, 1))
    outputs = r_k(inputs)
    assert outputs.shape == inputs.shape
