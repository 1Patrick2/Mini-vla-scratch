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


def _infer_keys(sample: Dict[str, Any]) -> Dict[str, str]:
    """Infer image/state/action/language keys from a sample dict."""
    keys: Dict[str, str] = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (1, 3):
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

    # Determine dimensions from the first sample
    state_val = first.get(state_key)
    action_val = first.get(action_key)

    state_dim = (
        int(state_val.shape[0])
        if isinstance(state_val, (np.ndarray, torch.Tensor, list))
        else 0
    )
    action_dim = (
        int(action_val.shape[0])
        if isinstance(action_val, (np.ndarray, torch.Tensor, list))
        else 0
    )

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
