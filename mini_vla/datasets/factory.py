"""Dataset factory — maps ``dataset_type`` to the appropriate dataset class."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter
from mini_vla.datasets.robot_adapter import BaseRobotDatasetAdapter
from mini_vla.datasets.toy_2d_dataset import Toy2DDataset


def build_dataset(
    data_cfg: Dict[str, Any],
    base_dataset: Sequence[Dict[str, Any]] | None = None,
) -> Toy2DDataset | PushTDatasetAdapter | BaseRobotDatasetAdapter:
    """Build a dataset from a config section.

    Args:
        data_cfg: The ``data`` subsection of the merged config (must contain
            ``dataset_type`` and type-specific keys).
        base_dataset: Optional pre-loaded list of sample dicts. If given,
            the adapter wraps them directly instead of loading data.

    Returns:
        A ``Toy2DDataset``, ``PushTDatasetAdapter``, or
        ``BaseRobotDatasetAdapter`` instance.

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

    if dtype == "robot_dataset":
        return _build_robot_dataset(data_cfg, base_dataset)

    raise ValueError(f"Unsupported dataset_type: '{dtype}'")


def _build_robot_dataset(
    data_cfg: Dict[str, Any],
    base_dataset: Sequence[Dict[str, Any]] | None = None,
) -> BaseRobotDatasetAdapter:
    """Build a ``BaseRobotDatasetAdapter`` from a ``robot_dataset`` config."""
    from mini_vla.datasets.normalization import ActionNormalizer, load_stats
    from mini_vla.datasets.registry import get_dataset_spec

    spec = get_dataset_spec(data_cfg["dataset_name"])

    if base_dataset is None:
        loader = data_cfg.get("loader", "lerobot")
        repo_id = data_cfg.get("repo_id", spec.repo_id)
        max_samples = data_cfg.get("max_samples", 0)

        if loader == "lerobot":
            from mini_vla.datasets.lerobot_loader import load_lerobot_samples
            local_root = data_cfg.get("local_root")
            base_dataset = load_lerobot_samples(
                repo_id=repo_id,
                root=local_root,
                max_samples=max_samples,
            )
        else:
            raise ValueError(f"Unsupported loader for robot_dataset: '{loader}'")

    # ── Split-aware filtering ────────────────────────────────────
    split_cfg = data_cfg.get("split", {})
    if split_cfg.get("enabled", False):
        from mini_vla.datasets.splits import filter_samples_by_episode, load_split

        manifest = load_split(split_cfg["path"])
        split_name = split_cfg.get("name", "train")
        episode_ids = (
            manifest.train_episode_ids if split_name == "train"
            else manifest.eval_episode_ids
        )
        n_before = len(base_dataset)
        base_dataset = filter_samples_by_episode(base_dataset, episode_ids)
        print(f"[factory] Split '{split_name}': {len(base_dataset)}/{n_before} samples kept")
        if not base_dataset:
            raise ValueError(
                f"Split '{split_name}' produced empty dataset. "
                f"Available episode ids: {manifest.train_episode_ids + manifest.eval_episode_ids}"
            )

    # Optional normalization
    normalizer = None
    norm_cfg = data_cfg.get("normalization", {})
    if norm_cfg.get("enabled", False):
        stats_path = Path(norm_cfg["stats_path"])
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalization.enabled=true but stats_path does not exist: {stats_path}. "
                "Run scripts/compute_dataset_stats.py first."
            )
        stats = load_stats(stats_path)
        normalizer = ActionNormalizer(stats)

    dataset = BaseRobotDatasetAdapter(
        base_dataset,
        spec=spec,
        image_size=data_cfg.get("image_size", 64),
        max_text_len=data_cfg.get("max_text_len", 16),
        normalizer=normalizer,
        strict=data_cfg.get("strict", True),
    )

    # Optional history transform
    hist_cfg = data_cfg.get("history", {})
    if hist_cfg.get("enabled", False):
        from mini_vla.datasets.transforms import HistoryDatasetWrapper
        dataset = HistoryDatasetWrapper(
            dataset,
            history_size=hist_cfg.get("history_size", 1),
            include_prev_state=hist_cfg.get("include_prev_state", True),
            include_prev_action=hist_cfg.get("include_prev_action", True),
            first_frame_prev_state=hist_cfg.get("first_frame_prev_state", "current"),
            first_frame_prev_action=hist_cfg.get("first_frame_prev_action", "zero"),
            concat_to_state=hist_cfg.get("concat_to_state", False),
            state_dim=data_cfg.get("state_dim", 2),
        )

    return dataset


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
