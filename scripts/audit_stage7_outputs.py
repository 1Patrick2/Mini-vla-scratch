"""Audit Stage 7 outputs for data-integrity and reproducibility.

Reads the split manifest, train-only stats metadata, and all three
eval reports, then validates 9 conditions that a strict benchmark must
satisfy.  Exits 0 on full pass, 1 on any failure.

Usage:
    python scripts/audit_stage7_outputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "outputs/splits/pusht_seed42_80_20_split.json"
STATS_META_PATH = (
    PROJECT_ROOT / "outputs/dataset_reports/pusht_strict_train_stats.meta.json"
)
REPORT_PATHS: List[Path] = [
    PROJECT_ROOT / "outputs/eval/pusht_strict_single_frame_report.json",
    PROJECT_ROOT / "outputs/eval/pusht_strict_history_report.json",
    PROJECT_ROOT / "outputs/eval/pusht_strict_delta_report.json",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _check(ok: bool, label: str, detail: str = "") -> int:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if detail:
        print(f"       {detail}")
    return 0 if ok else 1


def main() -> None:
    failures = 0

    print("=" * 60)
    print("  Stage 7 Audit")
    print("=" * 60)

    # ── 1. Load all inputs ──────────────────────────────────────────────
    split = _load_json(SPLIT_PATH)
    stats_meta = _load_json(STATS_META_PATH)
    reports: List[Dict[str, Any]] = [_load_json(p) for p in REPORT_PATHS]
    sf_report, hist_report, delta_report = reports  # noqa: F841

    train_ids = set(split.get("train_episode_ids", []))
    eval_ids = set(split.get("eval_episode_ids", []))

    # ── 2. Check 1: no overlap ──────────────────────────────────────────
    overlap = train_ids & eval_ids
    failures += _check(
        len(overlap) == 0,
        "train/eval episode ids have no overlap",
        f"overlap={overlap}" if overlap else "",
    )

    # ── 3. Check 2: stats meta split=train ──────────────────────────────
    failures += _check(
        stats_meta.get("split") == "train",
        "stats meta split=train",
        f"got={stats_meta.get('split')!r}",
    )

    # ── 4. Check 3: stats num_samples = 745 ─────────────────────────────
    actual_num = stats_meta.get("num_samples")
    # Accept the known value from real data: 745
    failures += _check(
        actual_num == 745,
        "stats train samples = 745",
        f"got={actual_num}",
    )

    # ── 5. Check 4: all reports use same split path ─────────────────────
    split_paths = {r.get("dataset", {}).get("split", {}).get("path") for r in reports}
    failures += _check(
        len(split_paths) == 1 and None not in split_paths,
        "all reports share the same split path",
        f"paths={split_paths}",
    )

    # ── 6. Check 5: all reports num_eval_samples = 279 ──────────────────
    eval_counts = {r.get("dataset", {}).get("num_eval_samples") for r in reports}
    failures += _check(
        eval_counts == {279},
        "all reports evaluate 279 eval samples",
        f"counts={eval_counts}",
    )

    # ── 7. Check 6: all reports eval_episode_ids match ──────────────────
    eval_ep_groups: List[frozenset] = []
    for r in reports:
        ids = r.get("dataset", {}).get("split", {}).get("eval_episode_ids", [])
        eval_ep_groups.append(frozenset(ids))
    failures += _check(
        len(set(eval_ep_groups)) == 1,
        "all reports have identical eval_episode_ids",
        f"ids={[sorted(ep) for ep in eval_ep_groups]}",
    )

    # ── 8. Check 7: delta report target type = delta_action ─────────────
    target_type = delta_report.get("dataset", {}).get("target", {}).get("type")
    failures += _check(
        target_type == "delta_action",
        "delta report target.type = delta_action",
        f"got={target_type!r}",
    )

    # ── 9. Check 8: delta reconstruction string ─────────────────────────
    reconstruction = (
        delta_report.get("dataset", {}).get("target", {}).get("reconstruction") or ""
    )
    failures += _check(
        "pred_action = prev_action + pred_delta" in reconstruction,
        "delta reconstruction equals pred_action = prev_action + pred_delta",
        f"got={reconstruction!r}",
    )

    # ── 10. Check 9: delta raw MAE < previous-action baseline raw MAE ───
    delta_raw_mae = delta_report.get("raw_action_metrics", {}).get("mae", float("inf"))
    prev_raw_mae = (
        delta_report.get("baselines", {})
        .get("previous_action", {})
        .get("mae", float("inf"))
    )
    failures += _check(
        delta_raw_mae < prev_raw_mae,
        "DeltaAction beats previous-action baseline",
        f"delta_mae={delta_raw_mae:.6f}  prev_mae={prev_raw_mae:.6f}",
    )

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    if failures == 0:
        print("  Stage 7 audit passed.")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"  Stage 7 audit: {failures} check(s) FAILED.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
