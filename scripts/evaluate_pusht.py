"""PushT evaluation CLI — offline evaluation with baselines.

Usage:
    python scripts/evaluate_pusht.py \
        --config configs/train/pusht_debug.yaml \
        --ckpt outputs/checkpoints/best.pt \
        --max-samples 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_vla.config import load_config
from mini_vla.datasets.factory import build_dataset
from mini_vla.evaluation.evaluator import evaluate_policy_on_dataset
from mini_vla.inference.predictor import Predictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a MiniVLA checkpoint on PushT data.",
    )
    parser.add_argument("--config", default="configs/train/pusht_debug.yaml",
                        help="Training config file path.")
    parser.add_argument("--ckpt", required=True,
                        help="Checkpoint path (e.g. outputs/checkpoints/best.pt).")
    parser.add_argument("--repo-id",
                        help="Override repo ID for PushT dataset.")
    parser.add_argument("--local-root",
                        help="Local PushT dataset root.")
    parser.add_argument("--max-samples", type=int, default=256,
                        help="Max samples to evaluate (default 256).")
    parser.add_argument("--output", default="outputs/eval/pusht_report.json",
                        help="Output report JSON path.")
    parser.add_argument("--predictions-output", default=None,
                        help="Optional JSONL path for per-sample predictions.")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference (default cpu).")
    parser.add_argument("--no-clip-action", action="store_true",
                        help="Disable action clipping.")
    parser.add_argument("--action-limit", type=float, default=0.05,
                        help="Action clip limit (default 0.05).")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_config(Path(args.config))

    # Build dataset (PushT adapter, mock or remote)
    data_cfg = config["data"]
    if args.repo_id:
        data_cfg["repo_id"] = args.repo_id
    dataset = build_dataset(data_cfg)

    # Build predictor
    predictor = Predictor(
        config,
        args.ckpt,
        device=args.device,
        clip_action=not args.no_clip_action,
        action_limit=args.action_limit,
    )

    # Evaluate
    print(f"Evaluating on up to {args.max_samples} samples ...")
    report = evaluate_policy_on_dataset(
        predictor,
        dataset,
        action_dim=config["model"]["action_dim"],
        max_samples=args.max_samples,
    )

    # Print summary
    m = report["model"]
    print("\nModel:")
    print(f"  MAE:    {m['mae']:.6f}")
    print(f"  MSE:    {m['mse']:.6f}")
    print(f"  RMSE:   {m['rmse']:.6f}")
    print(f"  CosSim: {m['cosine_similarity']:.4f}")
    print(f"  Finite: {m['finite_ratio']:.4f}")

    if "baselines" in report:
        print("\nBaselines:")
        for name, bm in report["baselines"].items():
            beats = "✓" if bm["mae"] > m["mae"] else "✗"
            print(f"  {name:20s}  MAE={bm['mae']:.6f}  ({beats} model beats baseline)")

    # Save report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_path}")

    # Save predictions
    if args.predictions_output:
        pred_path = Path(args.predictions_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for sample in dataset[: min(args.max_samples, len(dataset))]:
                pred = predictor.predict(sample)
                line = {
                    "pred_action": pred.tolist(),
                    "gt_action": sample["action"].tolist(),
                }
                f.write(json.dumps(line) + "\n")
        print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
