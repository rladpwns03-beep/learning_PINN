"""CLI entrypoint for training the PINN."""

from __future__ import annotations

import argparse
from typing import Sequence

from pinn.config import TrainConfig
from pinn.train import train_model


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    """Parse CLI arguments into a TrainConfig."""

    parser = argparse.ArgumentParser(description="Train a D-BJH-based PINN.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--num-collocation", type=int, default=512)
    parser.add_argument("--domain-lower", type=float, nargs="+", default=[0.0, 0.0])
    parser.add_argument("--domain-upper", type=float, nargs="+", default=[1.0, 1.0])

    args = parser.parse_args(argv)
    return TrainConfig(
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_layers=tuple(args.hidden_layers),
        num_collocation=args.num_collocation,
        domain_lower=tuple(args.domain_lower),
        domain_upper=tuple(args.domain_upper),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI main function."""

    config = parse_args(argv)
    train_model(config)


if __name__ == "__main__":
    main()
