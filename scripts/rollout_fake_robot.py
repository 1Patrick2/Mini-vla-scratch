"""Fake robot rollout entry point placeholder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out Mini VLA in a fake robot.")
    parser.add_argument("--ckpt", required=False)
    parser.parse_args()
    print("Fake robot rollout will be implemented after inference.")


if __name__ == "__main__":
    main()
