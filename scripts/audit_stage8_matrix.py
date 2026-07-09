"""Audit Stage 8 matrix outputs for completeness.

Supports two modes:
- ``structure-only`` (default): checks config structure, entry labels,
  but does NOT require report files to exist.
- ``full``: also requires all smoke reports and matrix summary files.

Usage:
    python scripts/audit_stage8_matrix.py --mode structure-only
    python scripts/audit_stage8_matrix.py --mode full
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Audit Stage 8 matrix outputs.")
    parser.add_argument("--mode", default="structure-only",
                        choices=["structure-only", "full"],
                        help="Audit mode: structure-only checks config, full also checks files")
    args = parser.parse_args()
    mode = args.mode

    failures = 0

    print("=" * 60)
    print(f"  Stage 8 Matrix Audit (mode: {mode})")
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

    # ── 3. PushT full benchmark entries exist in config ────────────────
    pusht_full = [
        e for e in entries
        if e.get("status") == "full_benchmark" and "pusht" in e.get("dataset", "")
    ]
    failures += _check(
        len(pusht_full) >= 3,
        f"PushT full benchmark entries in config: {len(pusht_full)}",
    )

    # ── 4. ActionChunk entry exists ────────────────────────────────────
    ac_entries = [e for e in entries if e.get("policy") == "action_chunk_bc"]
    failures += _check(
        len(ac_entries) >= 1,
        "ActionChunk entry exists in config",
    )

    # ── 5. ALOHA entries are marked smoke, not full_benchmark ──────────
    aloha_entries = [e for e in entries if "aloha" in e.get("dataset", "").lower()]
    for entry in aloha_entries:
        failures += _check(
            entry.get("status") != "full_benchmark",
            f"ALOHA {entry['policy']} is not marked full_benchmark",
            f"status={entry.get('status')}",
        )

    # ── 6. LIBERO is feasibility only ──────────────────────────────────
    libero = [e for e in entries if "libero" in e.get("dataset", "").lower()]
    for entry in libero:
        failures += _check(
            entry.get("status") == "feasibility",
            "LIBERO is feasibility only",
            f"status={entry.get('status')}",
        )

    # ── 7. Full-mode checks: report files must exist ───────────────────
    if mode == "full":
        for entry in entries:
            rp = entry.get("report", "")
            if not rp:
                continue
            report_path = Path(rp)
            exists = report_path.exists()
            failures += _check(
                exists,
                f"Report exists: {entry['dataset']} / {entry['policy']}",
                str(report_path) if not exists else "",
            )

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
