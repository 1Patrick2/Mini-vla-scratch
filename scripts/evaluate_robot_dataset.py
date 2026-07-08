"""Generic robot dataset evaluation CLI — supports normalization.

Usage:
    python scripts/evaluate_robot_dataset.py \
        --config configs/train/pusht_normalized.yaml \
        --ckpt outputs/checkpoints/best.pt \
        --dataset-name pusht
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from mini_vla.config import load_config
from mini_vla.datasets.factory import build_dataset
from mini_vla.datasets.normalization import ActionNormalizer, load_stats
from mini_vla.evaluation.action_metrics import cosine_similarity, finite_ratio, mae, mse, rmse
from mini_vla.evaluation.baselines import (
    MeanActionBaseline,
    PreviousActionBaseline,
    ZeroActionBaseline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a robot dataset.",
    )
    parser.add_argument("--config", required=True,
                        help="Training config file path.")
    parser.add_argument("--ckpt", required=True,
                        help="Checkpoint path.")
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset name (e.g. pusht).")
    parser.add_argument("--repo-id", default="",
                        help="Override repo ID.")
    parser.add_argument("--loader", default=None,
                        choices=["lerobot", "hf_datasets"],
                        help="Data loader backend.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to evaluate (default 0 = all).")
    parser.add_argument("--heldout-ratio", type=float, default=0.2,
                        help="Fraction of episodes to hold out.")
    parser.add_argument("--output", default="outputs/eval/robot_report.json",
                        help="Output report JSON path.")
    parser.add_argument("--predictions-output", default=None,
                        help="Optional JSONL path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock-data", action="store_true",
                        help="Use synthetic mock data.")
    parser.add_argument("--split-path", default=None,
                        help="Path to a split manifest JSON.")
    parser.add_argument("--split", default="eval", choices=["train", "eval", "all"],
                        help="Which split to evaluate (default eval).")
    return parser


def _build_mock_data(action_dim=2, n=16):
    """Generate mock samples for testing."""
    rng = np.random.RandomState(42)
    return [
        {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": rng.randn(2).astype(np.float32),
            "action": rng.randn(action_dim).astype(np.float32),
            "episode_index": i // 4,
            "frame_index": i % 4,
        }
        for i in range(n)
    ]


def _split_by_episode(samples, heldout_ratio=0.2):
    """Split by episode_index."""
    if heldout_ratio <= 0:
        return samples, samples
    episodes: Dict[int, List[dict]] = {}
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


def _compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    return {
        "mae": mae(preds, targets),
        "mse": mse(preds, targets),
        "rmse": rmse(preds, targets),
        "cosine_similarity": cosine_similarity(preds, targets),
        "finite_ratio": finite_ratio(preds),
    }


def _raw_action(sample: Dict[str, Any]) -> torch.Tensor:
    """Get the raw (un-normalised) action from a sample."""
    if "action_raw" in sample:
        return sample["action_raw"]
    return sample["action"]


def _get_action_views(
    sample: Dict[str, Any],
    pred: torch.Tensor,
    normalizer: Optional[ActionNormalizer],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(pred_norm, gt_norm, pred_raw, gt_raw)`` from a sample.

    ``pred`` is the model output (always in training space = normalized
    if normalizer is active, raw otherwise).
    """
    if normalizer:
        gt_norm = sample.get("action_normalized", sample["action"])
        gt_raw = sample.get("action_raw")
        if gt_raw is None:
            gt_raw = normalizer.denormalize_action(gt_norm)
        pred_norm = pred
        pred_raw = normalizer.denormalize_action(pred_norm)
    else:
        gt_raw = sample.get("action_raw", sample["action"])
        pred_raw = pred
        pred_norm = pred
        gt_norm = gt_raw
    return pred_norm, gt_norm, pred_raw, gt_raw


def _compute_mean_raw_action(samples: List[Dict[str, Any]]) -> torch.Tensor:
    """Compute the mean action in raw space from a list of samples."""
    actions = [_raw_action(s).clone().detach().cpu() for s in samples]
    if not actions:
        return torch.zeros(2)
    return torch.stack(actions).mean(dim=0)


def evaluate_samples_with_predictor(
    samples: List[Dict[str, Any]],
    predictor: Any,
    normalizer: Optional[ActionNormalizer] = None,
    train_samples_for_baseline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a predictor on samples and return metrics + predictions list.

    Args:
        samples: Evaluation samples.
        predictor: Object with ``predict(sample) -> Tensor``.
        normalizer: Optional normalizer for action views.
        train_samples_for_baseline: Training samples used to compute the
            mean baseline action.  If None, uses ``samples`` (eval pool),
            but this may cause slight data leakage.

    Returns:
        ``(report_metrics, predictions_jsonl)``.
    """
    pred_norm_list: List[torch.Tensor] = []
    gt_norm_list: List[torch.Tensor] = []
    pred_raw_list: List[torch.Tensor] = []
    gt_raw_list: List[torch.Tensor] = []
    preds_jsonl: List[Dict[str, Any]] = []

    for sample in samples:
        pred = predictor.predict(sample)
        pred_norm, gt_norm, pred_raw, gt_raw = _get_action_views(
            sample, pred, normalizer,
        )

        pred_norm_list.append(pred_norm.cpu().clone())
        gt_norm_list.append(gt_norm.cpu().clone())
        pred_raw_list.append(pred_raw.cpu().clone())
        gt_raw_list.append(gt_raw.cpu().clone())

        entry: Dict[str, Any] = {
            "pred_action_raw": pred_raw_list[-1].tolist(),
            "gt_action_raw": gt_raw_list[-1].tolist(),
            "pred_action_normalized": pred_norm_list[-1].tolist(),
            "gt_action_normalized": gt_norm_list[-1].tolist(),
        }
        ep = sample.get("episode_index")
        if ep is not None:
            entry["episode_index"] = int(ep)
        fi = sample.get("frame_index")
        if fi is not None:
            entry["frame_index"] = int(fi)
        preds_jsonl.append(entry)

    pred_raw_t = torch.stack(pred_raw_list)
    gt_raw_t = torch.stack(gt_raw_list)
    pred_norm_t = torch.stack(pred_norm_list)
    gt_norm_t = torch.stack(gt_norm_list)

    act_dim = pred_raw_t.shape[-1]

    report_metrics: Dict[str, Any] = {
        "num_eval_samples": len(samples),
        "action_dim": act_dim,
        "raw_action_metrics": _compute_metrics(pred_raw_t, gt_raw_t),
        "normalized_action_metrics": _compute_metrics(pred_norm_t, gt_norm_t),
    }

    # Baselines (raw action space)
    zero = ZeroActionBaseline(act_dim)
    zero_preds = torch.stack([zero.predict(s) for s in samples])
    bs = report_metrics.setdefault("baselines", {})
    bs["zero_action"] = _compute_metrics(zero_preds, gt_raw_t)

    # Mean baseline — use train samples to avoid data leakage
    baseline_source = train_samples_for_baseline or samples
    mean_act = _compute_mean_raw_action(baseline_source)
    mean_preds = torch.stack([MeanActionBaseline(mean_act).predict() for _ in samples])
    report_metrics["baselines"]["mean_action"] = _compute_metrics(mean_preds, gt_raw_t)

    prev = PreviousActionBaseline(act_dim)
    prev_preds = torch.stack([prev.predict(s) for s in samples])
    report_metrics["baselines"]["previous_action"] = _compute_metrics(prev_preds, gt_raw_t)

    return report_metrics, preds_jsonl


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    data_cfg = config["data"]

    # Build predictor
    from mini_vla.inference.predictor import Predictor
    predictor = Predictor(
        config, args.ckpt, device=args.device,
        clip_action=False,
    )

    # Load dataset
    if args.mock_data:
        mock_samples = _build_mock_data()
        from mini_vla.datasets.registry import get_dataset_spec
        from mini_vla.datasets.robot_adapter import BaseRobotDatasetAdapter
        spec = get_dataset_spec(args.dataset_name)
        dataset = BaseRobotDatasetAdapter(mock_samples, spec)
        eval_samples = [dataset[i] for i in range(len(dataset))]
        train_samples = []
    else:
        eval_cfg = copy.deepcopy(data_cfg) if args.split_path else data_cfg
        if args.split_path:
            eval_cfg["split"] = {"enabled": False}
        dataset = build_dataset(eval_cfg)
        n = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
        all_samples = [dataset[i] for i in range(n)]

        if args.split_path:
            from mini_vla.datasets.splits import filter_samples_by_episode, load_split
            manifest = load_split(args.split_path)
            if args.split == "train":
                eval_samples = filter_samples_by_episode(all_samples, manifest.train_episode_ids)
                train_samples = []
            elif args.split == "eval":
                eval_samples = filter_samples_by_episode(all_samples, manifest.eval_episode_ids)
                train_samples = filter_samples_by_episode(all_samples, manifest.train_episode_ids)
            else:
                eval_samples = all_samples
                train_samples = all_samples
        else:
            train_samples, eval_samples = _split_by_episode(all_samples, args.heldout_ratio)

    if args.split_path and args.split == "eval" and not eval_samples:
        raise ValueError(
            "Eval split produced empty dataset. "
            "Check split manifest or max_samples."
        )
    if args.split_path and args.split == "eval":
        pool = eval_samples
    elif args.heldout_ratio > 0 and len(eval_samples) > 0:
        pool = eval_samples
    else:
        pool = eval_samples if eval_samples else train_samples

    if not pool:
        print("Error: no samples to evaluate.", file=sys.stderr)
        sys.exit(1)

    # Set up normalizer if enabled
    normalizer: Optional[ActionNormalizer] = None
    norm_cfg = data_cfg.get("normalization", {})
    if norm_cfg.get("enabled", False):
        stats_path = Path(norm_cfg["stats_path"])
        if stats_path.exists():
            stats = load_stats(stats_path)
            normalizer = ActionNormalizer(stats)

    report_metrics, predictions = evaluate_samples_with_predictor(
        pool, predictor,
        normalizer=normalizer,
        train_samples_for_baseline=train_samples,
    )

    # Build full report
    report: Dict[str, Any] = {
        "dataset": {
            "dataset_name": args.dataset_name,
            "repo_id": args.repo_id or data_cfg.get("repo_id", ""),
            "loader": data_cfg.get("loader", "lerobot"),
            "num_eval_samples": len(pool),
            "num_train_samples": len(train_samples),
            "heldout_ratio": args.heldout_ratio,
            "has_image": True,
            "action_dim": report_metrics["action_dim"],
            "normalization": {
                "enabled": normalizer is not None,
                "stats_path": str(norm_cfg.get("stats_path", "")) if normalizer else None,
            },
            "split": {
                "enabled": bool(args.split_path),
                "path": args.split_path,
                "name": args.split,
                "seed": getattr(manifest, "seed", None) if args.split_path else None,
                "train_episode_ids": manifest.train_episode_ids if args.split_path else None,
                "eval_episode_ids": manifest.eval_episode_ids if args.split_path else None,
                "num_train_samples": len(train_samples),
                "num_eval_samples": len(pool),
                "stats_source": "train_split_only" if normalizer else None,
            } if args.split_path else None,
        },
        **report_metrics,
    }

    # Print
    m = report["raw_action_metrics"]
    print("\nRaw action metrics:")
    print(f"  MAE:    {m['mae']:.6f}")
    print(f"  MSE:    {m['mse']:.6f}")
    print(f"  RMSE:   {m['rmse']:.6f}")
    print(f"  CosSim: {m['cosine_similarity']:.4f}")
    print(f"  Finite: {m['finite_ratio']:.4f}")
    if "baselines" in report:
        print("\nBaselines (raw action):")
        for name, bm in report["baselines"].items():
            if not isinstance(bm, dict) or "mae" not in bm:
                continue
            beats = "Y" if bm["mae"] > m["mae"] else "N"
            print(f"  {name:20s}  MAE={bm['mae']:.6f}  ({beats} model beats baseline)")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out_path}")

    if args.predictions_output and predictions:
        pred_path = Path(args.predictions_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            for entry in predictions:
                f.write(json.dumps(entry) + "\n")
        print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
