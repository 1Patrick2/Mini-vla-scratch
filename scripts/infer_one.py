"""Single-sample inference entry point placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Mini VLA inference sample.")
    parser.add_argument("--ckpt", required=False)
    parser.add_argument("--image", required=False)
    parser.add_argument("--instruction", required=False)
    parser.parse_args()
    print("Inference will be implemented after model training.")


if __name__ == "__main__":
    main()
