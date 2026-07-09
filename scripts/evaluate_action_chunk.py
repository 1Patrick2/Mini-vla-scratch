"""Evaluate ActionChunk BC model — produces chunk-aware metrics.

Usage:
    python scripts/evaluate_action_chunk.py \\
        --config configs/experiments/pusht/action_chunk_bc_smoke.yaml \\
        --ckpt outputs/runs/pusht_action_chunk_smoke/checkpoints/best.pt \\
        --dataset-name pusht \\
        --split-path outputs/splits/pusht_seed42_80_20_split.json \\
        --split eval \\
        --output outputs/eval/pusht_action_chunk_smoke_report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from mini_vla.config import load_config
from mini_vla.datasets.factory import build_dataset
from mini_vla.evaluation.action_metrics import finite_ratio, mae, mse, rmse


def _compute_chunk_metrics(
    pred_chunks: List[torch.Tensor],
    gt_chunks: List[torch.Tensor],
) -> Dict[str, float]:
    pred_t = torch.stack(pred_chunks)  # [N, H, D]
    gt_t = torch.stack(gt_chunks)
    return {
        "mae_all": float(mae(pred_t, gt_t)),
        "mae_first": float(mae(pred_t[:, 0], gt_t[:, 0])),
        "mse_all": float(mse(pred_t, gt_t)),
        "rmse_all": float(rmse(pred_t, gt_t)),
        "finite_ratio": float(finite_ratio(pred_t)),
    }


def build_eval_samples(
    config_path: str,
    dataset_name: str,
    split_path: Optional[str] = None,
    split: str = "eval",
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    """Load evaluation samples from config."""
    config = load_config(Path(config_path))

    if split_path:
        from mini_vla.datasets.splits import filter_samples_by_episode, load_split
        data_cfg = copy.deepcopy(config["data"])
        data_cfg["split"] = {"enabled": False}
        dataset = build_dataset(data_cfg)
        n = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
        all_samples = [dataset[i] for i in range(n)]
        manifest = load_split(split_path)
        ep_ids = manifest.train_episode_ids if split == "train" else manifest.eval_episode_ids
        pool = filter_samples_by_episode(all_samples, ep_ids)
    else:
        data_cfg = config["data"]
        dataset = build_dataset(data_cfg)
        n = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
        pool = [dataset[i] for i in range(n)]

    if not pool:
        raise ValueError("Eval split produced empty dataset.")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ActionChunk BC model with chunk metrics."
    )
    parser.add_argument("--config", required=True, help="Training config path")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--split", default="eval", choices=["train", "eval", "all"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/eval/action_chunk_report.json")
    parser.add_argument("--predictions-output", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Build predictor (uses policy path for action_chunk_bc)
    from mini_vla.inference.predictor import Predictor
    predictor = Predictor(config, args.ckpt, device=args.device, clip_action=False)

    # Load eval samples
    pool = build_eval_samples(
        args.config, args.dataset_name,
        split_path=args.split_path, split=args.split,
        max_samples=args.max_samples,
    )

    # Evaluate
    pred_chunks: List[torch.Tensor] = []
    gt_chunks: List[torch.Tensor] = []
    predictions: List[Dict[str, Any]] = []

    for sample in pool:
        # Get gt action_chunk
        gt_chunk = sample.get("action_chunk")
        if gt_chunk is None:
            gt_chunk = sample.get("action", torch.zeros(2)).unsqueeze(0)

        gt_chunks.append(
            gt_chunk.cpu() if isinstance(gt_chunk, torch.Tensor)
            else torch.tensor(gt_chunk)
        )

        # Use Predictor.predict_dict (avoids accessing _policy directly)
        out = predictor.predict_dict(sample)
        pred_chunk = out.get("action_chunk", out["action"].unsqueeze(0))
        pred_chunks.append(pred_chunk)

        entry: Dict[str, Any] = {
            "pred_action_chunk": pred_chunks[-1].tolist(),
            "gt_action_chunk": gt_chunks[-1].tolist(),
        }
        ep = sample.get("episode_index")
        if ep is not None:
            entry["episode_index"] = int(ep)
        fi = sample.get("frame_index")
        if fi is not None:
            entry["frame_index"] = int(fi)
        predictions.append(entry)

    metrics = _compute_chunk_metrics(pred_chunks, gt_chunks)

    # Build report
    report: Dict[str, Any] = {
        "dataset": {
            "dataset_name": args.dataset_name,
            "repo_id": args.repo_id or config.get("data", {}).get("repo_id", ""),
            "num_eval_samples": len(pool),
            "target": {
                "type": "action_chunk",
                "action_horizon": config.get("shape_meta", {}).get("action_horizon", 4),
                "action_dim": config.get("shape_meta", {}).get("action", {}).get("dim", 2),
            },
        },
        "chunk_metrics": metrics,
    }

    # Print
    print("\nActionChunk metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

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
