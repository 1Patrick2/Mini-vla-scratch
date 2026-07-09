"""Predictor / Policy inference for MiniVLA.

Loads a trained checkpoint and provides ``predict()``, ``predict_dict()``,
and ``select_action()`` for single-sample inference.

Supports both legacy raw-MiniVLA checkpoints and new policy-based checkpoints,
including ActionChunk policies that return full chunk dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from mini_vla.models import build_model
from mini_vla.training.checkpoint import load_checkpoint


class Predictor:
    """Policy-style predictor for MiniVLA inference.

    Args:
        config: Model config dictionary (top-level or ``model`` section).
        checkpoint_path: Path to a ``.pt`` checkpoint file.
        device: Device to run inference on (default ``"cpu"``).
        clip_action: Whether to clip the output action (default ``True``).
        action_limit: Max absolute value per action dimension (default ``0.05``).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str | Path,
        device: str = "cpu",
        clip_action: bool = True,
        action_limit: float = 0.05,
    ) -> None:
        self.device = torch.device(device)
        self.clip_action = clip_action
        self.action_limit = action_limit
        self._policy = None

        # Try policy path first (new), fall back to raw model (legacy)
        if config.get("policy", {}).get("type"):
            from mini_vla.policies import build_policy
            self._policy = build_policy(config).to(self.device)
            self.model = self._policy.model
        else:
            self.model = build_model(config).to(self.device)

        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()

    # Optional tensor keys to pass through for delta/history reconstruction
    _OPTIONAL_KEYS = [
        "prev_action",
        "prev_action_raw",
        "prev_action_normalized",
        "prev_state",
        "prev_state_raw",
        "prev_state_normalized",
        "delta_action",
        "delta_action_raw",
        "delta_action_normalized",
    ]

    def _build_batch(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build a single-sample batch from a dataset sample dict."""
        batch: Dict[str, torch.Tensor] = {
            "image": sample["image"].unsqueeze(0).to(self.device),
            "input_ids": sample["input_ids"].unsqueeze(0).to(self.device),
            "state": sample["state"].unsqueeze(0).to(self.device),
        }
        if "attention_mask" in sample:
            batch["attention_mask"] = (
                sample["attention_mask"].unsqueeze(0).to(self.device)
            )
        # Pass through optional keys (needed for delta/history reconstruction)
        for key in self._OPTIONAL_KEYS:
            if key in sample and isinstance(sample[key], torch.Tensor):
                batch[key] = sample[key].unsqueeze(0).to(self.device)
        return batch

    def predict_dict(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run inference and return the full policy output dict.

        The returned dict depends on the policy type:

        * Single-step policies (SingleFrame, History, DeltaAction):
          ``{"action": Tensor[action_dim]}``
        * ActionChunk policy:
          ``{"action": Tensor[action_dim], "action_chunk": Tensor[H, action_dim]}``

        Args:
            sample: Single sample dict with keys ``image`` [3,H,W],
                ``input_ids`` [T], ``attention_mask`` [T], ``state`` [state_dim].

        Returns:
            Dictionary of output tensors on CPU.
        """
        batch = self._build_batch(sample)

        with torch.no_grad():
            if self._policy is not None:
                out = self._policy.predict_action(batch)
            else:
                pred = self.model(batch)
                out = {"action": pred}

        # Move to CPU and squeeze batch dim
        result: Dict[str, torch.Tensor] = {}
        for key, val in out.items():
            squeezed = val.squeeze(0).cpu()
            if key == "action" and self.clip_action:
                squeezed = torch.clamp(squeezed, -self.action_limit, self.action_limit)
            result[key] = squeezed

        return result

    def predict(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run inference and return the predicted action tensor.

        Convenience method equivalent to ``predict_dict(sample)["action"]``.

        Args:
            sample: Single sample dict.

        Returns:
            Tensor[action_dim] on CPU — the predicted action.
        """
        return self.predict_dict(sample)["action"]

    def select_action(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Policy-style API: select an action from an observation.

        Delegates to :meth:`predict`.

        Args:
            observation: Same format as a dataset sample dict.

        Returns:
            Tensor[action_dim] on CPU.
        """
        return self.predict(observation)


__all__ = ["Predictor"]
