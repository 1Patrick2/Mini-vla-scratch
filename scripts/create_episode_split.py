"""Create a reproducible episode-level dataset split manifest.

Usage:
    python scripts/create_episode_split.py --dataset-name pusht --mock-data --output split.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_vla.datasets.registry import get_dataset_spec
from mini_vla.datasets.splits import create_episode_split, save_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducible episode-level split manifest.",
    )
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset name (e.g. pusht).")
    parser.add_argument("--repo-id", default="",
                        help="Override Hugging Face repo ID.")
    parser.add_argument("--loader", default="lerobot",
                        choices=["lerobot", "hf_datasets"],
                        help="Data loader backend.")
    parser.add_argument("--max-samples", type=int, default=1024,
                        help="Max samples to load for episode discovery.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of episodes for training (default 0.8).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic shuffle.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path for the split manifest.")
    parser.add_argument("--mock-data", action="store_true",
                        help="Use synthetic mock data instead of real dataset.")
    return parser


def _make_mock_samples(spec, n=32):
    """Generate mock samples with multiple episodes."""
    import numpy as np
    rng = np.random.RandomState(42)
    samples = []
    for i in range(n):
        samples.append({
            spec.image_keys[0]: rng.randint(0, 256, (3, 96, 96), dtype=np.uint8)
            if spec.image_keys else None,
            spec.state_keys[0]: rng.randn(2).astype(np.float32) if spec.state_keys else None,
            spec.action_keys[0]: rng.randn(2).astype(np.float32) if spec.action_keys else None,
            "episode_index": i // 4,
            "frame_index": i % 4,
        })
    return samples


def main() -> None:
    args = build_parser().parse_args()

    spec = get_dataset_spec(args.dataset_name)

    if args.mock_data:
        samples = _make_mock_samples(spec, n=args.max_samples)
    else:
        from mini_vla.datasets.lerobot_loader import load_lerobot_samples
        samples = load_lerobot_samples(
            repo_id=args.repo_id or spec.repo_id,
            max_samples=args.max_samples,
        )

    split = create_episode_split(
        samples,
        dataset_name=args.dataset_name,
        repo_id=args.repo_id or spec.repo_id,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    out_path = save_split(split, args.output)
    print(f"Split saved to {out_path}")
    print(f"  train episodes: {len(split.train_episode_ids)} "
          f"({split.num_train_samples} samples)")
    print(f"  eval episodes:  {len(split.eval_episode_ids)} "
          f"({split.num_eval_samples} samples)")


if __name__ == "__main__":
    main()
