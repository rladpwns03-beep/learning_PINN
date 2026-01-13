"""Neural network definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for shared MLP structure."""

    hidden_layers: Sequence[int]
    normalize_input: bool = True
    enforce_positive_output: bool = True
    output_log: bool = False
    input_min: float = 0.0
    input_max: float = 1.0


class ThetaNet(tf.keras.Model):
    """Predicts theta values from relative pressure inputs."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _build_mlp(input_dim=1, output_dim=1, hidden_layers=config.hidden_layers)
        self.softplus = tf.keras.layers.Activation("softplus")

    def call(self, pressure: tf.Tensor) -> tf.Tensor:
        inputs = _normalize_input(pressure, self.config)
        outputs = self.network(inputs)
        if self.config.enforce_positive_output:
            outputs = self.softplus(outputs)
        return outputs


class LNet(tf.keras.Model):
    """Predicts non-negative L values from radius inputs."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _build_mlp(input_dim=1, output_dim=1, hidden_layers=config.hidden_layers)
        self.softplus = tf.keras.layers.Activation("softplus")

    def call(self, radius: tf.Tensor) -> tf.Tensor:
        inputs = _normalize_input(radius, self.config)
        outputs = self.network(inputs)
        if self.config.output_log:
            outputs = tf.exp(outputs)
        else:
            outputs = self.softplus(outputs)
        return outputs


def dtheta_dp(theta_net: ThetaNet, pressure: tf.Tensor) -> tf.Tensor:
    """Compute dTheta/dp via autograd."""

    pressure = tf.convert_to_tensor(pressure)
    with tf.GradientTape() as tape:
        tape.watch(pressure)
        theta = theta_net(pressure)
    grads = tape.gradient(theta, pressure)
    return grads


def _build_mlp(input_dim: int, output_dim: int, hidden_layers: Sequence[int]) -> tf.keras.Sequential:
    layers: list[tf.keras.layers.Layer] = []
    for width in hidden_layers:
        layers.append(tf.keras.layers.Dense(width, activation="tanh", input_shape=(input_dim,)))
        input_dim = width
    layers.append(tf.keras.layers.Dense(output_dim))
    return tf.keras.Sequential(layers)


def _normalize_input(inputs: tf.Tensor, config: NetworkConfig) -> tf.Tensor:
    if not config.normalize_input:
        return inputs
    denom = max(config.input_max - config.input_min, 1e-8)
    return (inputs - config.input_min) / denom
