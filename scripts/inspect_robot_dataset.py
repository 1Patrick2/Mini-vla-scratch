"""Generic robot dataset inspect CLI.

Usage:
    python scripts/inspect_robot_dataset.py --dataset-name pusht --mock-data
    python scripts/inspect_robot_dataset.py \\
        --dataset-name pusht --repo-id lerobot/pusht --loader lerobot \\
        --max-samples 8 --output outputs/dataset_reports/pusht_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_vla.datasets.robot_inspection import inspect_robot_dataset_from_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a robot dataset schema via DatasetSpec.",
    )
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset name (e.g. pusht, aloha_sim_transfer_cube).")
    parser.add_argument("--repo-id", default="",
                        help="Override Hugging Face repo ID.")
    parser.add_argument("--loader", default="lerobot",
                        choices=["lerobot", "hf_datasets"],
                        help="Data loader backend (default lerobot).")
    parser.add_argument("--max-samples", type=int, default=8,
                        help="Max samples to load (default 8).")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: stdout).")
    parser.add_argument("--mock-data", action="store_true",
                        help="Use synthetic mock data instead of real dataset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    samples = None
    if not args.mock_data:
        try:
            from mini_vla.datasets.lerobot_loader import load_lerobot_samples
            from mini_vla.datasets.registry import get_dataset_spec
            spec = get_dataset_spec(args.dataset_name)
            samples = load_lerobot_samples(
                repo_id=args.repo_id or spec.repo_id,
                max_samples=args.max_samples,
                validate=False,
            )
        except (ImportError, Exception) as e:
            print(f"[ERROR] Could not load real data: {e}", file=sys.stderr)
            print("Use --mock-data for offline inspection.", file=sys.stderr)
            sys.exit(1)

    try:
        report = inspect_robot_dataset_from_registry(
            args.dataset_name,
            samples=samples,
            repo_id=args.repo_id,
            loader=args.loader,
        )
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

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
