"""Predictor / Policy inference for MiniVLA.

Loads a trained checkpoint and provides ``predict()`` and ``select_action()``
for single-sample inference following LeRobot-style policy API.
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
        clip_action_flag: bool = True,
        action_limit: float = 0.05,
    ) -> None:
        self.device = torch.device(device)
        self.model = build_model(config).to(self.device)
        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()
        self.clip_action_flag = clip_action_flag
        self.action_limit = action_limit

    def predict(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run inference on a single dataset sample.

        Automatically batchifies the sample (adds a batch dimension),
        runs the model under ``torch.no_grad()``, and returns a 1-D CPU
        action tensor, optionally clipped to ``[-action_limit, action_limit]``.

        Args:
            sample: Single sample dict with keys ``image`` [3,H,W],
                ``input_ids`` [T], ``attention_mask`` [T], ``state`` [2].
                Extra keys (``action``, ``instruction``, etc.) are ignored.

        Returns:
            Tensor[action_dim] on CPU — the predicted action.
        """
        # Build a batch of size 1
        batch: Dict[str, torch.Tensor] = {
            "image": sample["image"].unsqueeze(0).to(self.device),
            "input_ids": sample["input_ids"].unsqueeze(0).to(self.device),
            "state": sample["state"].unsqueeze(0).to(self.device),
        }
        if "attention_mask" in sample:
            batch["attention_mask"] = (
                sample["attention_mask"].unsqueeze(0).to(self.device)
            )

        with torch.no_grad():
            action_pred = self.model(batch)  # Tensor[1, action_dim]

        action = action_pred.squeeze(0).cpu()  # Tensor[action_dim]

        if self.clip_action_flag:
            action = torch.clamp(action, -self.action_limit, self.action_limit)

        return action

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
