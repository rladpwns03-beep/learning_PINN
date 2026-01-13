import tensorflow as tf

from d_bjh_pinn.physics.volterra import (
    VolterraConfig,
    build_volterra_operator,
    cumsum_volterra,
    masked_matmul_volterra,
)


def _naive_volterra(
    pressure: tf.Tensor,
    r_grid: tf.Tensor,
    l_values: tf.Tensor,
    dr: tf.Tensor,
    kernel: tf.Tensor,
    j_idx: tf.Tensor,
) -> tf.Tensor:
    outputs = []
    for i in range(int(pressure.shape[0])):
        total = tf.constant(0.0, dtype=pressure.dtype)
        for k in range(int(j_idx[i].numpy()), int(r_grid.shape[0])):
            total = total + kernel[i, k] * l_values[k] * dr[k]
        outputs.append(total)
    return tf.stack(outputs)


def test_volterra_matches_naive() -> None:
    pressure = tf.constant([0.1, 0.3], dtype=tf.float32)
    r_grid = tf.constant([0.2, 0.4, 0.6], dtype=tf.float32)
    l_values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    dr = tf.constant([0.1, 0.1, 0.1], dtype=tf.float32)
    j_idx = tf.constant([0, 1], dtype=tf.int32)

    kernel = build_volterra_operator(VolterraConfig(kernel_scale=2.0))(pressure, r_grid)
    expected = _naive_volterra(pressure, r_grid, l_values, dr, kernel, j_idx)

    masked = masked_matmul_volterra(pressure, r_grid, l_values, dr, kernel, j_idx)
    cumsum = cumsum_volterra(pressure, r_grid, l_values, dr, kernel, j_idx)

    tf.debugging.assert_near(masked, expected)
    tf.debugging.assert_near(cumsum, expected)


def test_masked_matmul_gradcheck() -> None:
    pressure = tf.constant([0.2, 0.5], dtype=tf.float64)
    r_grid = tf.constant([0.1, 0.3, 0.7], dtype=tf.float64)
    dr = tf.constant([0.1, 0.1, 0.1], dtype=tf.float64)
    j_idx = tf.constant([0, 1], dtype=tf.int32)

    def kernel(p: tf.Tensor, r: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
        return scale * (p[:, None] + r[None, :])

    def func(l_values: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
        return masked_matmul_volterra(
            pressure,
            r_grid,
            l_values,
            dr,
            lambda p, r: kernel(p, r, scale),
            j_idx,
        )

    l_values = tf.Variable([1.0, 2.0, 3.0], dtype=tf.float64)
    scale = tf.Variable(1.5, dtype=tf.float64)
    (dy_l, dy_scale), (num_l, num_scale) = tf.test.compute_gradient(func, [l_values, scale])
    tf.debugging.assert_near(dy_l, num_l, atol=1e-4, rtol=1e-4)
    tf.debugging.assert_near(dy_scale, num_scale, atol=1e-4, rtol=1e-4)
