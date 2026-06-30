"""Training command entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_vla.config import load_config
from mini_vla.training.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Mini VLA.")
    parser.add_argument(
        "--config",
        default="configs/train/debug.yaml",
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print trainer summary without training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    trainer = Trainer(config)
    print(trainer.describe())
    if not args.dry_run:
        print("Stage 0 skeleton only: training loop will be implemented in Stage 3.")


if __name__ == "__main__":
    main()
