"""Action and state normalisation for robot datasets.

Computes normalisation statistics from dataset samples and provides
a normaliser that can be applied during training and reversed during
evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class NormalizationStats:
    """Normalisation statistics for state and action tensors.

    Attributes:
        state_mean: Per-dimension mean of the state.
        state_std: Per-dimension standard deviation of the state.
        action_mean: Per-dimension mean of the action.
        action_std: Per-dimension standard deviation of the action.
        eps: Small constant to avoid division by zero.
    """

    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    eps: float = 1e-6


class ActionNormalizer:
    """Normalise and denormalise state/action tensors.

    Args:
        stats: A ``NormalizationStats`` instance.
    """

    def __init__(self, stats: NormalizationStats) -> None:
        self.stats = stats

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalise state to zero-mean unit-variance."""
        return (state - self.stats.state_mean) / (self.stats.state_std + self.stats.eps)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalise action to zero-mean unit-variance."""
        return (action - self.stats.action_mean) / (self.stats.action_std + self.stats.eps)

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        """Convert a normalised action back to the original scale."""
        return (
            action_norm * (self.stats.action_std + self.stats.eps)
            + self.stats.action_mean
        )


def compute_stats(
    samples: Sequence[Dict[str, Any]],
    state_key: str = "observation.state",
    action_key: str = "action",
    state_keys: Optional[Sequence[str]] = None,
    action_keys: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
) -> NormalizationStats:
    """Compute mean and std for state and action from a list of samples.

    Supports candidate key lists via ``state_keys`` and ``action_keys``.
    If candidate lists are provided, the first matching key from each
    sample is used (via ``find_first_key``).  Falls back to the
    single-key params for backward compatibility.

    Args:
        samples: List of sample dicts containing state and action keys.
        state_key: Single key fallback (default ``observation.state``).
        action_key: Single key fallback (default ``action``).
        state_keys: Optional list of candidate state keys.
        action_keys: Optional list of candidate action keys.
        eps: Small constant for std denominator.

    Returns:
        A ``NormalizationStats`` instance.
    """
    from mini_vla.datasets.key_utils import find_first_key

    state_list: List[torch.Tensor] = []
    action_list: List[torch.Tensor] = []
    for s in samples:
        state_k = find_first_key(s, state_keys) if state_keys else state_key
        action_k = find_first_key(s, action_keys) if action_keys else action_key
        st = s.get(state_k) if isinstance(state_k, str) else None
        ac = s.get(action_k) if isinstance(action_k, str) else None
        if st is None and state_keys:
            # Try all candidates directly as flat keys
            for k in state_keys:
                if k in s:
                    st = s[k]
                    break
        if ac is None and action_keys:
            for k in action_keys:
                if k in s:
                    ac = s[k]
                    break
        if st is not None:
            state_list.append(torch.as_tensor(st, dtype=torch.float32))
        if ac is not None:
            action_list.append(torch.as_tensor(ac, dtype=torch.float32))

    def _mean_std(tensors: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(tensors)
        return stacked.mean(dim=0), stacked.std(dim=0, correction=0)

    state_mean, state_std = (
        _mean_std(state_list) if state_list else (torch.tensor([]), torch.tensor([]))
    )
    action_mean, action_std = (
        _mean_std(action_list) if action_list else (torch.tensor([]), torch.tensor([]))
    )

    return NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        eps=eps,
    )


def save_stats(stats: NormalizationStats, path: str | Path) -> Path:
    """Save normalisation stats to a JSON file.

    Args:
        stats: The stats to save.
        path: Output JSON path.

    Returns:
        The resolved ``Path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_mean": stats.state_mean.tolist(),
        "state_std": stats.state_std.tolist(),
        "action_mean": stats.action_mean.tolist(),
        "action_std": stats.action_std.tolist(),
        "eps": stats.eps,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_stats(path: str | Path) -> NormalizationStats:
    """Load normalisation stats from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        A ``NormalizationStats`` instance.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    return NormalizationStats(
        state_mean=torch.tensor(payload["state_mean"], dtype=torch.float32),
        state_std=torch.tensor(payload["state_std"], dtype=torch.float32),
        action_mean=torch.tensor(payload["action_mean"], dtype=torch.float32),
        action_std=torch.tensor(payload["action_std"], dtype=torch.float32),
        eps=float(payload.get("eps", 1e-6)),
    )


__all__ = [
    "ActionNormalizer",
    "NormalizationStats",
    "compute_stats",
    "load_stats",
    "save_stats",
]
