"""Compute normalisation statistics for a robot dataset.

Usage:
    python scripts/compute_dataset_stats.py --dataset-name pusht --mock-data
    python scripts/compute_dataset_stats.py \\
        --dataset-name pusht --repo-id lerobot/pusht --loader lerobot \\
        --max-samples 256 --output outputs/dataset_reports/pusht_stats.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from mini_vla.datasets.normalization import compute_stats, save_stats
from mini_vla.datasets.registry import get_dataset_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute normalisation stats for a robot dataset.",
    )
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset name (e.g. pusht).")
    parser.add_argument("--repo-id", default="",
                        help="Override Hugging Face repo ID.")
    parser.add_argument("--loader", default="lerobot",
                        choices=["lerobot", "hf_datasets"],
                        help="Data loader backend (default lerobot).")
    parser.add_argument("--max-samples", type=int, default=256,
                        help="Max samples to use for stats (default 256).")
    parser.add_argument("--output", default="outputs/dataset_reports/stats.json",
                        help="Output JSON path.")
    parser.add_argument("--mock-data", action="store_true",
                        help="Use synthetic mock data instead of real dataset.")
    return parser


def _make_mock_samples(spec, n=64):
    """Generate mock samples for stats computation."""
    rng = np.random.RandomState(42)
    samples = []
    for _ in range(n):
        s = {}
        if spec.state_keys:
            s[spec.state_keys[0]] = rng.randn(2).astype(np.float32)
        if spec.action_keys:
            s[spec.action_keys[0]] = rng.randn(2).astype(np.float32)
        samples.append(s)
    return samples


def main() -> None:
    args = build_parser().parse_args()

    spec = get_dataset_spec(args.dataset_name)

    if args.mock_data:
        samples = _make_mock_samples(spec, n=args.max_samples)
    else:
        try:
            from mini_vla.datasets.lerobot_loader import load_lerobot_samples
            samples = load_lerobot_samples(
                repo_id=args.repo_id or spec.repo_id,
                max_samples=args.max_samples,
            )
        except (ImportError, Exception) as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

    stats = compute_stats(
        samples,
        state_keys=spec.state_keys,
        action_keys=spec.action_keys,
    )

    out_path = save_stats(stats, args.output)
    print(f"Stats saved to {out_path}")
    print(f"  state_mean: {stats.state_mean.tolist()}")
    print(f"  state_std:  {stats.state_std.tolist()}")
    print(f"  action_mean: {stats.action_mean.tolist()}")
    print(f"  action_std:  {stats.action_std.tolist()}")


if __name__ == "__main__":
    main()
