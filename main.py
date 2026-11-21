"""CLI entry point for training, evaluating, and running the Fusion Transformer demo."""

import argparse
import math
from typing import Optional, Sequence

from config import Config
from src.pipeline import Pipeline
from src.utils import setup_logger


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser supporting --train, --evaluate, and --demo modes."""
    parser = argparse.ArgumentParser(
        description="Multimodal Fusion Transformer (FT-Transformer + Text Cross-Attention)"
    )
    parser.add_argument("--train", action="store_true", help="Train and save model checkpoint")
    parser.add_argument(
        "--evaluate", action="store_true", help="Load saved checkpoint and evaluate on test split"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run end-to-end training, evaluation, and sample predictions (default if no flag)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="AdamW learning rate")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/fusion_transformer.pt",
        help="Path to save/load model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=None,
        help=(
            "Exit non-zero when the run's macro F1 is below this threshold in "
            "[0, 1], so training/evaluation can gate a CI job."
        ),
    )
    return parser


def exit_code_for(metrics: dict, min_macro_f1: Optional[float]) -> int:
    """Map reported metrics to a process exit code under a macro F1 gate.

    Returns 0 when no threshold is given or the run's macro F1 meets it, and 1
    when it falls below. The threshold must be a finite value in [0, 1];
    anything else raises ``ValueError`` so a typo cannot silently disable gating.
    """
    if min_macro_f1 is None:
        return 0
    if not math.isfinite(min_macro_f1) or not 0.0 <= min_macro_f1 <= 1.0:
        raise ValueError(
            "min_macro_f1 must be a finite value in [0, 1]; "
            f"got {min_macro_f1!r}"
        )
    return 1 if metrics["macro_f1"] < min_macro_f1 else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute requested train, evaluate, or demo workflow."""
    setup_logger()
    args = build_parser().parse_args(argv)
    config = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    )
    pipeline = Pipeline(config)

    run_demo = args.demo or (not args.train and not args.evaluate)
    if run_demo:
        metrics = pipeline.demo()
        print(
            f"\nDemo Complete: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )
        return exit_code_for(metrics, args.min_macro_f1)

    if args.train:
        summary = pipeline.train()
        m = summary.test_metrics
        print(
            f"\nTraining Complete (checkpoint={summary.checkpoint_path}): "
            f"test_acc={m['accuracy']:.4f}, precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, macro_f1={m['macro_f1']:.4f}, roc_auc={m['roc_auc']:.4f}"
        )

    if args.evaluate:
        m = pipeline.evaluate()
        print(
            f"\nEvaluation Complete: loss={m['loss']:.4f}, accuracy={m['accuracy']:.4f}, "
            f"precision={m['precision']:.4f}, recall={m['recall']:.4f}, "
            f"macro_f1={m['macro_f1']:.4f}, roc_auc={m['roc_auc']:.4f}"
        )

    return exit_code_for(m if args.evaluate else summary.test_metrics, args.min_macro_f1)


if __name__ == "__main__":
    raise SystemExit(main())
