"""Inspect a PushT-like dataset schema and statistics.

Usage:
    python scripts/inspect_pusht_dataset.py --repo-id lerobot/pusht --max-samples 128
    python scripts/inspect_pusht_dataset.py --local-root /path/to/pusht --max-samples 64
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect PushT dataset schema and statistics.",
    )
    parser.add_argument("--repo-id", default="lerobot/pusht",
                        help="Hugging Face dataset repo ID (default lerobot/pusht).")
    parser.add_argument("--local-root",
                        help="Local path to the dataset (optional, overrides repo-id).")
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Max samples to load for stats (default 128).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: stdout).")
    return parser


def _load_samples(
    repo_id: str,
    local_root: str | None,
    max_samples: int,
) -> list[dict]:
    """Load PushT samples from Hugging Face datasets or mock data.

    Returns a list of sample dicts.

    Raises:
        ImportError: If ``datasets`` is not installed and remote loading
            is attempted.
    """
    if local_root is not None:
        # Local load — try lerobot format
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(repo_id, root=local_root)
            return [ds[i] for i in range(min(max_samples, len(ds)))]
        except ImportError:
            raise ImportError(
                "Local LeRobot dataset loading requires 'lerobot'. "
                "Install it with: pip install lerobot"
            ) from None

    # Remote load via Hugging Face datasets
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Remote dataset loading requires 'datasets'. "
            "Install it with: pip install datasets"
        ) from None

    hf_ds = load_dataset(repo_id, split="train", streaming=True)
    samples = []
    for i, row in enumerate(hf_ds):
        if max_samples > 0 and i >= max_samples:
            break
        samples.append(row)
    return samples


def main() -> None:
    args = build_parser().parse_args()

    try:
        samples = _load_samples(args.repo_id, args.local_root, args.max_samples)
    except ImportError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    report = build_pusht_report(samples, repo_id=args.repo_id, max_samples=args.max_samples)

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
