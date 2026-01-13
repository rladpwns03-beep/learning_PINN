import tensorflow as tf

from d_bjh_pinn.physics.mapping import r_p
from d_bjh_pinn.physics.tcurve import ThicknessConfig


def test_r_p_shape() -> None:
    pressure = tf.zeros((4, 1))
    values = r_p(pressure, thickness=ThicknessConfig())
    assert values.shape == pressure.shape
