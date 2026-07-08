"""Compare multiple evaluation reports into a markdown table.

Usage:
    python scripts/compare_eval_reports.py \
        --reports outputs/eval/report1.json outputs/eval/report2.json \
        --output outputs/eval/comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _beats(m: float, b: float) -> str:
    return "Y" if m < b else "N"


def _get_baseline_mae(baselines: dict, name: str) -> float:
    b = baselines.get(name, {})
    return b.get("mae", float("inf")) if isinstance(b, dict) else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluation reports and produce a markdown table."
    )
    parser.add_argument("--reports", nargs="+", required=True,
                        help="Paths to eval report JSON files.")
    parser.add_argument("--output", default=None,
                        help="Output markdown path (default stdout).")
    args = parser.parse_args()

    # Clean display names for method comparison
    name_map = {
        "single_frame": "SingleFrame",
        "history": "History",
        "delta": "DeltaAction",
    }

    rows: List[Dict[str, Any]] = []
    for path in args.reports:
        report = load_report(path)
        raw = report.get("raw_action_metrics", {})
        norm = report.get("normalized_action_metrics", {})
        baselines = report.get("baselines", {})

        zero_mae = _get_baseline_mae(baselines, "zero_action")
        mean_mae = _get_baseline_mae(baselines, "mean_action")
        prev_mae = _get_baseline_mae(baselines, "previous_action")
        model_mae = raw.get("mae", 0)

        raw_method = Path(path).stem.replace("pusht_strict_", "").replace("_report", "")
        method = name_map.get(raw_method, raw_method)
        rows.append({
            "method": method,
            "raw_mae": model_mae,
            "raw_rmse": raw.get("rmse", 0),
            "norm_mae": norm.get("mae", 0),
            "zero": zero_mae,
            "mean": mean_mae,
            "prev": prev_mae,
            "beats_zero": _beats(model_mae, zero_mae),
            "beats_mean": _beats(model_mae, mean_mae),
            "beats_prev": _beats(model_mae, prev_mae),
        })

    # Build table
    lines = [
        "| Method | Raw MAE | Raw RMSE | Norm MAE | Zero | Mean | Prev |"
        " BeatsZ | BeatsM | BeatsP |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']:12s} | {r['raw_mae']:.4f} | {r['raw_rmse']:.4f} "
            f"| {r['norm_mae']:.4f} | {r['zero']:.2f} | {r['mean']:.2f} "
            f"| {r['prev']:.2f} | {r['beats_zero']:>8s} | {r['beats_mean']:>9s} "
            f"| {r['beats_prev']:>9s} |"
        )

    output = "\n".join(lines)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Comparison saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
