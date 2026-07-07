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

import numpy as np

from mini_vla.config import load_config
from mini_vla.datasets.factory import build_dataset
from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter
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
    parser.add_argument("--heldout-ratio", type=float, default=0.0,
                        help="Fraction of episodes to hold out (default 0 = all).")
    parser.add_argument("--mock-data", action="store_true",
                        help="Use synthetic mock data instead of real PushT.")
    return parser


def _build_mock_eval_pipeline(args, config):
    """Build predictor + eval_samples from mock Toy2D data (no remote deps)."""
    from scripts.generate_toy_data import generate_toy_data

    data_root = generate_toy_data(
        output_root=Path(args.output).parent / "mock_data",
        num_episodes=2, max_steps=4, image_size=64, seed=42,
    )
    from mini_vla.datasets import Toy2DDataset
    ds = Toy2DDataset(root=data_root)
    raw_list = []
    for s in [ds[i] for i in range(min(8, len(ds)))]:
        raw_list.append({
            "observation.image": (s["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            "observation.state": s["state"].numpy(),
            "action": s["action"].numpy(),
            "episode_index": 0,
            "frame_index": 0,
        })
    adapter = PushTDatasetAdapter(raw_list)
    samples = [adapter[i] for i in range(len(adapter))]

    predictor = Predictor(
        config, args.ckpt, device=args.device,
        clip_action=not args.no_clip_action, action_limit=args.action_limit,
    )
    return predictor, samples


def _split_by_episode(samples, heldout_ratio=0.2):
    """Split by episode_index to avoid frame-level leakage."""
    if heldout_ratio <= 0:
        return samples, samples
    episodes = {}
    for s in samples:
        ep = s.get("episode_index", 0)
        episodes.setdefault(ep, []).append(s)
    sorted_eps = sorted(episodes.keys())
    n_hold = max(1, int(len(sorted_eps) * heldout_ratio))
    hold_eps = set(sorted_eps[-n_hold:])
    train, eval_ = [], []
    for ep, frames in episodes.items():
        if ep in hold_eps:
            eval_.extend(frames)
        else:
            train.extend(frames)
    return train, eval_


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    action_dim = config["model"]["action_dim"]

    if args.mock_data:
        predictor, eval_samples = _build_mock_eval_pipeline(args, config)
        train_samples = []
    else:
        data_cfg = config["data"]
        if args.repo_id:
            data_cfg["repo_id"] = args.repo_id
        if args.local_root:
            data_cfg["local_root"] = args.local_root
        dataset = build_dataset(data_cfg)

        predictor = Predictor(
            config, args.ckpt, device=args.device,
            clip_action=not args.no_clip_action, action_limit=args.action_limit,
        )

        n = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
        all_samples = [dataset[i] for i in range(n)]
        train_samples, eval_samples = _split_by_episode(
            all_samples, heldout_ratio=args.heldout_ratio,
        )

    if args.heldout_ratio > 0 and len(eval_samples) > 0:
        print(f"Evaluating on {len(eval_samples)} held-out samples "
              f"(train: {len(train_samples)}) ...")
        report = evaluate_policy_on_dataset(
            predictor, eval_samples,
            action_dim=action_dim,
            baseline_dataset=train_samples if train_samples else None,
            sort_by_episode_frame=True,
        )
    else:
        pool = eval_samples or []
        print(f"Evaluating on {len(pool)} samples (no held-out split) ...")
        report = evaluate_policy_on_dataset(
            predictor, pool, action_dim=action_dim,
            sort_by_episode_frame=True,
        )

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
            if not isinstance(bm, dict) or "mae" not in bm:
                continue
            beats = "Y" if bm["mae"] > m["mae"] else "N"
            print(f"  {name:20s}  MAE={bm['mae']:.6f}  ({beats} model beats baseline)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_path}")

    if args.predictions_output and len(eval_samples) > 0:
        pred_path = Path(args.predictions_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for sample in eval_samples:
                pred = predictor.predict(sample)
                f.write(json.dumps({
                    "pred_action": pred.tolist(),
                    "gt_action": sample["action"].tolist(),
                }) + "\n")
        print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
