"""Single-sample inference CLI for MiniVLA.

Usage:
    python scripts/infer_one.py \
        --config configs/train/debug.yaml \
        --ckpt outputs/checkpoints/best.pt \
        --sample-index 0 \
        --output outputs/predictions/prediction.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from mini_vla.config import load_config
from mini_vla.datasets import Toy2DDataset
from mini_vla.inference.predictor import Predictor
from mini_vla.inference.visualizer import save_prediction_visualization
from mini_vla.training.metrics import l1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Mini VLA inference sample and save visualization.",
    )
    parser.add_argument("--config", default="configs/train/debug.yaml",
                        help="Training config file path.")
    parser.add_argument("--ckpt", required=True,
                        help="Checkpoint path (e.g. outputs/checkpoints/best.pt).")
    parser.add_argument("--data-root",
                        help="Data root (overrides config value).")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index of the sample in the dataset (default 0).")
    parser.add_argument("--output", default="outputs/predictions/prediction.png",
                        help="Output PNG path.")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference (default cpu).")
    parser.add_argument("--arrow-scale", type=float, default=4.0,
                        help="Arrow length multiplier (default 4.0).")
    parser.add_argument("--no-clip-action", action="store_true",
                        help="Disable action clipping.")
    parser.add_argument("--action-limit", type=float, default=0.05,
                        help="Action clip limit (default 0.05).")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load config
    config = load_config(Path(args.config))
    if args.data_root:
        config["data"]["data_root"] = args.data_root

    data_root = config["data"]["data_root"]
    device = args.device

    # Build Predictor
    predictor = Predictor(
        config,
        args.ckpt,
        device=device,
        clip_action=not args.no_clip_action,
        action_limit=args.action_limit,
    )

    # Load dataset and get sample
    ds = Toy2DDataset(root=data_root)
    sample = ds[args.sample_index]

    # Predict
    pred_action = predictor.predict(sample)
    gt_action = sample.get("action")

    # Compute L1 error
    if gt_action is not None:
        error = l1(pred_action, gt_action)
    else:
        error = None

    # Print results
    print(f"Sample index:  {args.sample_index}")
    print(f"Pred action:   [{pred_action[0]:.6f}, {pred_action[1]:.6f}]")
    if gt_action is not None:
        print(f"GT action:     [{gt_action[0]:.6f}, {gt_action[1]:.6f}]")
        print(f"L1 error:      {error:.6f}")
    else:
        print("GT action:     (not available)")
    print(f"Output:        {args.output}")

    # Save visualization
    save_prediction_visualization(
        args.output,
        image=sample["image"],
        state=sample["state"],
        pred_action=pred_action,
        gt_action=gt_action,
        arrow_scale=args.arrow_scale,
    )
    print("Visualization saved.")


if __name__ == "__main__":
    main()
