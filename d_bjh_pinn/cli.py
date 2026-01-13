"""CLI entrypoints for training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import csv
import tensorflow as tf

from d_bjh_pinn.config import DataConfig, ModelConfig, TrainConfig
from d_bjh_pinn.eval import compute_metrics
from d_bjh_pinn.train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-BJH PINN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the PINN.")
    train_parser.add_argument("--csv", required=True)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--outdir", type=str, default="runs/run1")
    train_parser.add_argument("--use-lbfgs", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate metrics.")
    eval_parser.add_argument("--l-values", required=True)

    return parser


def _run_train(args: argparse.Namespace) -> None:
    data_config = DataConfig(csv_path=args.csv)
    model_config = ModelConfig()
    train_config = TrainConfig(
        epochs=args.epochs,
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        log_every=args.log_every,
    )
    outdir = Path(args.outdir)
    train(data_config, model_config, train_config, outdir=outdir, use_lbfgs=args.use_lbfgs)


def _run_eval(args: argparse.Namespace) -> None:
    values: list[float] = []
    with open(args.l_values, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                values.append(float(row[0]))
    l_values = tf.convert_to_tensor(values, dtype=tf.float32)
    metrics = compute_metrics(l_values)
    print(metrics)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        _run_train(args)
    if args.command == "eval":
        _run_eval(args)


if __name__ == "__main__":
    main()
