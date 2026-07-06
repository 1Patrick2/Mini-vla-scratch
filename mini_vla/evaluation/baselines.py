"""Baseline policies for offline evaluation comparison."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class ZeroActionBaseline:
    """Always predicts ``[0, 0, ...]``."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def predict(self, sample: Dict[str, Any]) -> torch.Tensor:
        return torch.zeros(self.action_dim)

    def __repr__(self) -> str:
        return f"ZeroActionBaseline(dim={self.action_dim})"


class MeanActionBaseline:
    """Always predicts the mean action computed from training data."""

    def __init__(self, mean_action: torch.Tensor) -> None:
        self.mean_action = mean_action.clone().detach().cpu()

    def predict(self, sample: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        return self.mean_action.clone()

    def __repr__(self) -> str:
        return f"MeanActionBaseline(mean={self.mean_action.tolist()})"


class PreviousActionBaseline:
    """Predicts the previous frame's expert action.

    For the first frame of each episode, falls back to zero.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self._prev: Optional[torch.Tensor] = None
        self._prev_episode: Optional[int] = None

    def reset(self) -> None:
        """Reset the internal state (call at episode boundaries)."""
        self._prev = None
        self._prev_episode = None

    def predict(self, sample: Dict[str, Any]) -> torch.Tensor:
        episode = sample.get("episode_index")
        if episode is not None and self._prev_episode is not None and episode != self._prev_episode:
            self._prev = None
        if self._prev is not None:
            result = self._prev.clone()
        else:
            result = torch.zeros(self.action_dim)
        # Store ground-truth action for the next frame
        if "action" in sample:
            self._prev = sample["action"].clone().detach().cpu()
        self._prev_episode = episode
        return result

    def __repr__(self) -> str:
        return "PreviousActionBaseline"


def compute_mean_action(dataset, max_samples: int = 0) -> torch.Tensor:
    """Compute the mean action over a dataset (or its first N samples).

    Args:
        dataset: Iterable of sample dicts that contain ``"action"``.
        max_samples: If > 0, limit to this many samples.

    Returns:
        Tensor[action_dim] — the mean action.
    """
    actions = []
    for i, sample in enumerate(dataset):
        if max_samples > 0 and i >= max_samples:
            break
        actions.append(sample["action"].clone().detach().cpu())
    if not actions:
        return torch.zeros(2)
    return torch.stack(actions).mean(dim=0)


__all__ = [
    "MeanActionBaseline",
    "PreviousActionBaseline",
    "ZeroActionBaseline",
    "compute_mean_action",
]
