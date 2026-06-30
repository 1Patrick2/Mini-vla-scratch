"""Evaluation entry point placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mini VLA.")
    parser.add_argument("--config", default="configs/train/debug.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.parse_args()
    print("Evaluation will be implemented after the training loop.")


if __name__ == "__main__":
    main()
