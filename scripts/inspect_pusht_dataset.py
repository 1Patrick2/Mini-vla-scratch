"""Inspect a PushT-like dataset schema and statistics.

Usage:
    python scripts/inspect_pusht_dataset.py --repo-id lerobot/pusht --loader lerobot --max-samples 8
    python scripts/inspect_pusht_dataset.py --repo-id lerobot/pusht \
        --loader hf_datasets --max-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_vla.datasets.pusht_inspection import build_pusht_report

REQUIRED_KEYS_FOR_VISION = ["observation.image", "observation.state", "action"]
IMAGE_KEY_CANDIDATES = [
    "observation.image",
    "observation.images.top",
    "observation.images.main",
    "image",
]


def find_image_key(sample: dict) -> str | None:
    for key in IMAGE_KEY_CANDIDATES:
        if key in sample:
            return key
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect PushT dataset schema and statistics.",
    )
    parser.add_argument("--repo-id", default="lerobot/pusht",
                        help="Hugging Face dataset repo ID (default lerobot/pusht).")
    parser.add_argument("--local-root",
                        help="Local path to the dataset (optional).")
    parser.add_argument("--loader", default="lerobot",
                        choices=["lerobot", "hf_datasets"],
                        help="Data loader backend (default lerobot).")
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Max samples to load for stats (default 128).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: stdout).")
    return parser


def _load_samples(
    repo_id: str,
    local_root: str | None,
    loader: str,
    max_samples: int,
) -> list[dict]:
    """Load samples using the chosen loader."""
    if loader == "lerobot":
        from mini_vla.datasets.pusht_lerobot_loader import load_pusht_lerobot
        return load_pusht_lerobot(
            repo_id=repo_id, root=local_root, max_samples=max_samples,
        )
    elif loader == "hf_datasets":
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("hf_datasets loader requires 'datasets'.") from None
        hf_ds = load_dataset(repo_id, split="train", streaming=True)
        samples = []
        for i, row in enumerate(hf_ds):
            if max_samples > 0 and i >= max_samples:
                break
            samples.append(row)
        return samples
    else:
        raise ValueError(f"Unknown loader: {loader}")


def main() -> None:
    args = build_parser().parse_args()

    try:
        samples = _load_samples(args.repo_id, args.local_root, args.loader, args.max_samples)
    except (ImportError, KeyError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Build standard report
    report = build_pusht_report(samples, repo_id=args.repo_id, max_samples=args.max_samples)

    # Add loader-specific fields
    matched_key = find_image_key(samples[0]) if samples else None
    report["loader"] = args.loader
    report["has_image"] = matched_key is not None
    report["matched_image_key"] = matched_key

    if args.loader == "hf_datasets" and not matched_key:
        report["recommendation"] = "Use loader=lerobot for vision evaluation."

    output = json.dumps(report, indent=2, default=str)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        print(f"Report saved to {path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
