"""Audit Stage 8 matrix outputs for completeness.

Checks:
1. Stage 7 audit still passes
2. Matrix config is readable
3. PushT full benchmark reports (3) exist
4. ActionChunk smoke report exists
5. ALOHA smoke reports (3) exist
6. ALOHA entries are marked as smoke, not full_benchmark
7. LIBERO is feasibility only
8. Matrix summary JSON and MD exist

Usage:
    python scripts/audit_stage8_matrix.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_CONFIG = PROJECT_ROOT / "configs/matrix/stage8_policy_dataset_matrix.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs/matrix"
SUMMARY_JSON = OUTPUT_DIR / "stage8_matrix_summary.json"
SUMMARY_MD = OUTPUT_DIR / "stage8_matrix_summary.md"


def _check(ok: bool, label: str, detail: str = "") -> int:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if detail:
        print(f"       {detail}")
    return 0 if ok else 1


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    failures = 0

    print("=" * 60)
    print("  Stage 8 Matrix Audit")
    print("=" * 60)

    # ── 1. Stage 7 audit still passes ──────────────────────────────────
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/audit_stage7_outputs.py")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    failures += _check(
        result.returncode == 0,
        "Stage 7 audit still passes",
    )

    # ── 2. Matrix config exists and is readable ────────────────────────
    config_exists = MATRIX_CONFIG.exists()
    failures += _check(config_exists, f"Matrix config exists: {MATRIX_CONFIG}")
    if config_exists:
        config = yaml.safe_load(MATRIX_CONFIG.read_text()) if yaml else {}
        entries: List[Dict[str, Any]] = config.get("matrix", [])
        failures += _check(len(entries) > 0, "Matrix config has entries")
    else:
        entries = []

    # ── 3. PushT full benchmark reports exist ──────────────────────────
    pusht_reports = [
        e for e in entries
        if e.get("status") == "full_benchmark" and "pusht" in e.get("dataset", "")
    ]
    for entry in pusht_reports:
        rp = entry.get("report", "")
        report_path = Path(rp) if rp else None
        exists = report_path and report_path.exists()
        failures += _check(
            exists,
            f"PushT report exists: {entry['policy']}",
            str(report_path) if report_path else "",
        )

    # ── 4. PushT action_chunk report exists ────────────────────────────
    ac_entries = [e for e in entries if e.get("policy") == "action_chunk_bc"]
    if ac_entries:
        rp = ac_entries[0].get("report", "")
        report_path = Path(rp) if rp else None
        exists = report_path and report_path.exists()
        failures += _check(
            exists,
            f"ActionChunk smoke report exists: {rp}",
        )

    # ── 5. ALOHA smoke reports exist ───────────────────────────────────
    aloha_entries = [e for e in entries if "aloha" in e.get("dataset", "").lower()]
    for entry in aloha_entries:
        rp = entry.get("report", "")
        report_path = Path(rp) if rp else None
        exists = report_path and report_path.exists()
        failures += _check(
            exists or entry.get("status") != "smoke",
            f"ALOHA report {'exists' if exists else 'missing'}: {entry['policy']}",
            str(report_path) if report_path else "",
        )

    # ── 6. ALOHA entries must be smoke, not full_benchmark ─────────────
    for entry in aloha_entries:
        failures += _check(
            entry.get("status") != "full_benchmark",
            f"ALOHA {entry['policy']} is not marked full_benchmark",
            f"status={entry.get('status')}",
        )

    # ── 7. LIBERO is feasibility only ──────────────────────────────────
    libero = [e for e in entries if "libero" in e.get("dataset", "").lower()]
    for entry in libero:
        failures += _check(
            entry.get("status") == "feasibility",
            "LIBERO is feasibility only",
            f"status={entry.get('status')}",
        )

    # ── 8. Matrix summary files exist ──────────────────────────────────
    failures += _check(SUMMARY_JSON.exists(), "Matrix summary JSON exists")
    failures += _check(SUMMARY_MD.exists(), "Matrix summary MD exists")

    # ── Summary ────────────────────────────────────────────────────────
    print("=" * 60)
    if failures == 0:
        print("  Stage 8 matrix audit passed.")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"  Stage 8 matrix audit: {failures} check(s) FAILED.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
