"""Training command entry point.

Usage:
    python scripts/train.py --config configs/train/debug.yaml
    python scripts/train.py --config configs/train/debug.yaml --dry-run
"""

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
        help="Load config and print summary without training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))

    model_name = config["model"]["name"]
    dataset_type = config["data"]["dataset_type"]
    epochs = config["train"]["epochs"]
    device = config["train"]["device"]

    print(f"Config: {args.config}")
    print(f"Model:  {model_name}")
    print(f"Data:   {dataset_type}")
    if dataset_type == "pusht":
        repo_id = config["data"].get("repo_id", "lerobot/pusht")
        max_samples = config["data"].get("max_samples", "all")
        print(f"  Repo:  {repo_id}")
        print(f"  Max:   {max_samples} samples")
    elif dataset_type == "robot_dataset":
        dataset_name = config["data"].get("dataset_name", "")
        repo_id = config["data"].get("repo_id", "")
        loader = config["data"].get("loader", "lerobot")
        max_samples = config["data"].get("max_samples", "all")
        split_name = config["data"].get("split", {}).get("name", "")
        print(f"  Dataset: {dataset_name}")
        print(f"  Repo:    {repo_id}")
        print(f"  Loader:  {loader}")
        print(f"  Max:     {max_samples} samples")
        if split_name:
            print(f"  Split:   {split_name}")
    else:
        data_root = config["data"]["data_root"]
        print(f"  Root:  {data_root}")
    print(f"Train:  {epochs} epochs, device={device}")

    if args.dry_run:
        print("Dry-run mode.  No training executed.")
        return

    # Validate data exists (Toy2D only)
    if dataset_type == "toy_2d":
        data_path = Path(config["data"]["data_root"])
        if not (data_path / "episodes").is_dir():
            print(
                f"Error: no data found at {config['data']['data_root']}.\n"
                f"Generate data first:\n"
                f"  python scripts/generate_toy_data.py "
                f"--config configs/data/toy_2d.yaml --num-episodes 5"
            )
            sys.exit(1)

    trainer = Trainer(config)
    print(f"\nStarting training for {epochs} epoch(s) ...")
    trainer.fit(epochs=epochs)

    ckpt_dir = trainer.checkpoint_dir
    print(f"\nDone!  Checkpoints saved to {ckpt_dir}/")
    print(f"  last: {ckpt_dir / 'last.pt'}")
    print(f"  best: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
