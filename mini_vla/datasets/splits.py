"""Episode-level dataset split manifest for reproducible train/eval splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class EpisodeSplit:
    """Reproducible episode-level train/eval split.

    Attributes:
        dataset_name: Dataset name (e.g. ``"pusht"``).
        repo_id: Hugging Face repo ID.
        seed: Random seed used for shuffling episodes.
        train_episode_ids: Episode indices assigned to training.
        eval_episode_ids: Episode indices assigned to evaluation.
        split_by: How episodes were split (default ``"episode_index"``).
        train_ratio: Fraction of episodes used for training.
        eval_ratio: Fraction of episodes used for evaluation.
        num_train_samples: Total samples in training split.
        num_eval_samples: Total samples in evaluation split.
    """

    dataset_name: str
    repo_id: str
    seed: int
    train_episode_ids: List[int] = field(default_factory=list)
    eval_episode_ids: List[int] = field(default_factory=list)
    split_by: str = "episode_index"
    train_ratio: float = 0.8
    eval_ratio: float = 0.2
    num_train_samples: int = 0
    num_eval_samples: int = 0


def collect_episode_ids(samples: Sequence[Dict[str, Any]]) -> List[int]:
    """Collect unique episode indices from a list of samples, sorted."""
    ids: set[int] = set()
    for s in samples:
        ep = s.get("episode_index")
        if ep is not None:
            ids.add(int(ep))
    return sorted(ids)


def create_episode_split(
    samples: Sequence[Dict[str, Any]],
    dataset_name: str,
    repo_id: str = "",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> EpisodeSplit:
    """Create a reproducible episode-level train/eval split.

    Episodes are shuffled deterministically by the given seed, then split
    by ``train_ratio``.  The resulting ``EpisodeSplit`` contains non-overlapping
    ``train_episode_ids`` and ``eval_episode_ids``.

    Args:
        samples: List of sample dicts containing ``episode_index``.
        dataset_name: Dataset name.
        repo_id: Hugging Face repo ID.
        train_ratio: Fraction of episodes for training (default 0.8).
        seed: Random seed for deterministic shuffle.

    Returns:
        An ``EpisodeSplit`` instance.
    """
    import random

    episode_ids = collect_episode_ids(samples)
    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    n_train = max(1, int(len(episode_ids) * train_ratio))
    train_ids = sorted(episode_ids[:n_train])
    eval_ids = sorted(episode_ids[n_train:])

    train_count = sum(1 for s in samples if s.get("episode_index") in train_ids)
    eval_count = sum(1 for s in samples if s.get("episode_index") in eval_ids)

    return EpisodeSplit(
        dataset_name=dataset_name,
        repo_id=repo_id,
        seed=seed,
        train_episode_ids=train_ids,
        eval_episode_ids=eval_ids,
        train_ratio=train_ratio,
        eval_ratio=1.0 - train_ratio,
        num_train_samples=train_count,
        num_eval_samples=eval_count,
    )


def filter_samples_by_episode(
    samples: Sequence[Dict[str, Any]],
    episode_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    """Filter samples, keeping only those whose ``episode_index`` is in ``episode_ids``."""
    id_set = set(episode_ids)
    return [s for s in samples if s.get("episode_index") in id_set]


def save_split(split: EpisodeSplit, path: str | Path) -> Path:
    """Save an ``EpisodeSplit`` to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(split)
    payload["created_by"] = "mini_vla.datasets.splits"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_split(path: str | Path) -> EpisodeSplit:
    """Load an ``EpisodeSplit`` from a JSON file."""
    path = Path(path)
    payload = json.loads(path.read_text())
    return EpisodeSplit(**{k: v for k, v in payload.items() if k != "created_by"})


def validate_split(split: EpisodeSplit) -> None:
    """Validate an ``EpisodeSplit``.

    Raises:
        ValueError: If there is overlap between train/eval episode ids,
            or if any required field is invalid.
    """
    train_set = set(split.train_episode_ids)
    eval_set = set(split.eval_episode_ids)
    overlap = train_set & eval_set
    if overlap:
        raise ValueError(
            f"Train/eval episode id overlap: {sorted(overlap)}"
        )
    if not split.train_episode_ids:
        raise ValueError("Train episode list is empty.")
    if not split.eval_episode_ids:
        raise ValueError("Eval episode list is empty.")


__all__ = [
    "EpisodeSplit",
    "collect_episode_ids",
    "create_episode_split",
    "filter_samples_by_episode",
    "load_split",
    "save_split",
    "validate_split",
]
