"""Dataset factory — maps ``dataset_type`` to the appropriate dataset class."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter
from mini_vla.datasets.toy_2d_dataset import Toy2DDataset


def build_dataset(
    data_cfg: Dict[str, Any],
    base_dataset: Sequence[Dict[str, Any]] | None = None,
) -> Toy2DDataset | PushTDatasetAdapter:
    """Build a dataset from a config section.

    Args:
        data_cfg: The ``data`` subsection of the merged config (must contain
            ``dataset_type`` and type-specific keys).
        base_dataset: Optional pre-loaded list of PushT-like samples. If
            given, the adapter wraps them directly instead of loading data.

    Returns:
        A ``Toy2DDataset`` or ``PushTDatasetAdapter`` instance.

    Raises:
        ValueError: If ``dataset_type`` is unsupported or loader is invalid.
        ImportError: If required optional dependency is missing.
    """
    dtype = data_cfg.get("dataset_type", "toy_2d")

    if dtype == "toy_2d":
        return Toy2DDataset(root=data_cfg["data_root"])

    if dtype == "pusht":
        if base_dataset is None:
            base_dataset = _load_pusht(data_cfg)
        return PushTDatasetAdapter(
            base_dataset,
            image_key=data_cfg.get("image_key", "observation.image"),
            image_keys=data_cfg.get("image_keys"),
            state_key=data_cfg.get("state_key", "observation.state"),
            action_key=data_cfg.get("action_key", "action"),
            image_size=data_cfg.get("image_size", 64),
            max_text_len=data_cfg.get("max_text_len", 16),
            instruction=data_cfg.get("instruction"),
        )

    raise ValueError(f"Unsupported dataset_type: '{dtype}'")


def _load_pusht(data_cfg: Dict[str, Any]) -> list[dict]:
    """Load PushT samples using the configured loader.

    Loader is chosen from ``data_cfg.get("loader", "lerobot")``.
    """
    loader = data_cfg.get("loader", "lerobot")
    repo_id = data_cfg.get("repo_id", "lerobot/pusht")
    max_samples = data_cfg.get("max_samples", 0)

    if loader == "lerobot":
        return _load_lerobot(repo_id, data_cfg, max_samples)
    elif loader == "hf_datasets":
        return _load_hf_datasets(repo_id, max_samples)
    else:
        raise ValueError(f"Unsupported PushT loader: '{loader}'")


def _load_lerobot(
    repo_id: str,
    data_cfg: Dict[str, Any],
    max_samples: int,
) -> list[dict]:
    """Load PushT samples with real images via LeRobotDataset."""
    from mini_vla.datasets.pusht_lerobot_loader import load_pusht_lerobot

    local_root = data_cfg.get("local_root")
    return load_pusht_lerobot(
        repo_id=repo_id,
        root=local_root,
        max_samples=max_samples,
    )


def _load_hf_datasets(repo_id: str, max_samples: int) -> list[dict]:
    """Load PushT samples via Hugging Face ``datasets`` (state/action only).

    This path does **not** provide ``observation.image``.  It is suitable
    for schema inspection but **not** for full vision training.

    Raises:
        ImportError: If ``datasets`` is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "hf_datasets PushT loading requires 'datasets'. "
            "Install it with: pip install datasets"
        ) from None

    hf_ds = load_dataset(repo_id, split="train", streaming=True)
    samples = []
    for i, row in enumerate(hf_ds):
        if max_samples > 0 and i >= max_samples:
            break
        samples.append(row)
    return samples


__all__ = ["build_dataset"]
