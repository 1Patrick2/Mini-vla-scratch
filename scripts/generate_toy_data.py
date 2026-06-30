"""Toy data generation entry point placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Toy 2D data.")
    parser.add_argument("--config", default="configs/data/toy_2d.yaml")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.parse_args()
    print("Stage 1 will implement toy data generation.")


if __name__ == "__main__":
    main()
