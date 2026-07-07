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
        ValueError: If ``dataset_type`` is unsupported.
    """
    dtype = data_cfg.get("dataset_type", "toy_2d")

    if dtype == "toy_2d":
        return Toy2DDataset(root=data_cfg["data_root"])

    if dtype == "pusht":
        if base_dataset is None:
            # Lazy remote-load (may raise ImportError)
            base_dataset = _load_pusht_remote(data_cfg)
        return PushTDatasetAdapter(
            base_dataset,
            image_key=data_cfg.get("image_key", "observation.image"),
            state_key=data_cfg.get("state_key", "observation.state"),
            action_key=data_cfg.get("action_key", "action"),
            image_size=data_cfg.get("image_size", 64),
            max_text_len=data_cfg.get("max_text_len", 16),
            instruction=data_cfg.get("instruction"),
        )

    raise ValueError(f"Unsupported dataset_type: '{dtype}'")


def _load_pusht_remote(data_cfg: Dict[str, Any]) -> list[dict]:
    """Load PushT samples from Hugging Face datasets or local LeRobot format.

    Raises:
        ImportError: If required dependencies are not installed.
    """
    local_root = data_cfg.get("local_root")
    repo_id = data_cfg.get("repo_id", "lerobot/pusht")
    max_samples = data_cfg.get("max_samples", 0)

    # Local LeRobot format
    if local_root:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(repo_id, root=local_root)
            n = max_samples if max_samples > 0 else len(ds)
            return [ds[i] for i in range(min(n, len(ds)))]
        except ImportError:
            raise ImportError(
                "Local LeRobot dataset loading requires 'lerobot'. "
                "Install it with: pip install lerobot"
            ) from None

    # Remote via Hugging Face datasets
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "PushT dataset loading requires 'datasets'. "
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
