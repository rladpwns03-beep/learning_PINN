"""Utility helpers for deterministic behavior."""

from __future__ import annotations

import random

import tensorflow as tf


def seed_everything(seed: int) -> None:
    """Set deterministic behavior for Python and torch."""

    random.seed(seed)
    tf.random.set_seed(seed)
