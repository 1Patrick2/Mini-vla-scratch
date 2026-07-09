"""Inspect a LeRobot dataset's shape metadata — image key, state/action dims.

Usage:
    python scripts/inspect_shape_meta.py \
        --dataset-name aloha_sim_transfer_cube \
        --repo-id lerobot/aloha_sim_transfer_cube_scripted \
        --loader lerobot \
        --max-samples 128 \
        --output outputs/dataset_reports/aloha_sim_shape_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


def _load_samples(
    dataset_name: str,
    repo_id: str,
    loader: str,
    max_samples: int,
) -> list[Dict[str, Any]]:
    """Load samples from a LeRobot dataset."""
    from mini_vla.datasets.registry import get_dataset_spec

    spec = get_dataset_spec(dataset_name)

    if loader == "lerobot":
        from mini_vla.datasets.lerobot_loader import load_lerobot_samples

        return load_lerobot_samples(
            repo_id=repo_id or spec.repo_id,
            max_samples=max_samples,
        )
    else:
        raise ValueError(f"Unsupported loader: '{loader}'")


def _infer_dim(x: Any) -> int:
    """Safely get the first dimension of state/action-like data.

    Handles ``torch.Tensor``, ``np.ndarray``, and ``list``.
    """
    if isinstance(x, torch.Tensor):
        return int(x.numel()) if x.ndim == 1 else int(x.shape[-1])
    if isinstance(x, np.ndarray):
        return int(x.size) if x.ndim == 1 else int(x.shape[-1])
    if isinstance(x, (list, tuple)):
        return len(x)
    return 0


def _is_image_like(v: Any) -> bool:
    """Check whether *v* is a plausible image tensor/array.

    Supports ``np.ndarray`` HWC/CHW and ``torch.Tensor`` CHW/HWC with
    1 or 3 channels.
    """
    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (1, 3):
        return True
    if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] in (1, 3):
        return True
    return False


def _infer_keys(sample: Dict[str, Any]) -> Dict[str, str]:
    """Infer image/state/action/language keys from a sample dict."""
    keys: Dict[str, str] = {}
    for k, v in sample.items():
        if _is_image_like(v):
            keys["image_key"] = k
        elif k.endswith(".state") or k == "state":
            keys["state_key"] = k
        elif k == "action" or k.endswith("action"):
            keys["action_key"] = k
        elif isinstance(v, str):
            keys["language_key"] = k
    return keys


def inspect_shape_meta(
    dataset_name: str,
    repo_id: str = "",
    loader: str = "lerobot",
    max_samples: int = 128,
) -> Dict[str, Any]:
    """Inspect a dataset and return shape metadata.

    Returns a dict with dataset_name, repo_id, image_key, state_key,
    action_key, language_key, state_dim, action_dim, num_samples_checked.
    """
    samples = _load_samples(dataset_name, repo_id, loader, max_samples)

    if not samples:
        raise ValueError(f"No samples loaded for {dataset_name}")

    first = samples[0]
    keys = _infer_keys(first)

    state_key = keys.get("state_key", "observation.state")
    action_key = keys.get("action_key", "action")
    image_key = keys.get("image_key")

    # Determine dimensions from the first sample using safe _infer_dim
    state_dim = _infer_dim(first.get(state_key))
    action_dim = _infer_dim(first.get(action_key))

    language_key = keys.get("language_key")

    result = {
        "dataset_name": dataset_name,
        "repo_id": repo_id,
        "state_key": state_key,
        "action_key": action_key,
        "image_key": image_key,
        "language_key": language_key,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "num_samples_checked": len(samples),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect shape metadata from a LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset-name", required=True,
        help="Dataset name (e.g. aloha_sim_transfer_cube)",
    )
    parser.add_argument("--repo-id", default="", help="Override repo ID")
    parser.add_argument("--loader", default="lerobot", choices=["lerobot"])
    parser.add_argument("--max-samples", type=int, default=128, help="Max samples to check")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = inspect_shape_meta(
        dataset_name=args.dataset_name,
        repo_id=args.repo_id,
        loader=args.loader,
        max_samples=args.max_samples,
    )

    print(f"\nShape metadata for {args.dataset_name}:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
