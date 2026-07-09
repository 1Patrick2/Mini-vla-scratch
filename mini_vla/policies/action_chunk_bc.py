"""ActionChunk BC policy — predict a chunk of future actions.

Input:  image + instruction + state (or history-augmented state)
Output: action_chunk: Tensor[B, action_horizon, action_dim]

The underlying MiniVLA model outputs a flattened vector of shape
``[B, action_horizon * action_dim]``, which the policy reshapes
into ``[B, action_horizon, action_dim]``.

``action_horizon`` is read from ``shape_meta.action_horizon`` (default 4).
``action_dim`` is read from ``shape_meta.action.dim`` (per-step dimension).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from mini_vla.datasets.shape_meta import get_action_dim, get_action_horizon
from mini_vla.policies.base import BasePolicy
from mini_vla.policies.registry import register_policy


class ActionChunkBCPolicy(BasePolicy):
    """Action Chunking Behavior Cloning policy.

    Predicts a chunk of future actions in a single forward pass.
    The model's ``action_dim`` must equal ``action_horizon * per_step_action_dim``
    (flattened output).  The policy reshapes internally.
    """

    policy_type = "action_chunk_bc"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.action_horizon = get_action_horizon(config)
        self.step_action_dim = get_action_dim(config)
        self._flattened_dim = self.action_horizon * self.step_action_dim

    def _reshape_chunk(self, pred_flat: torch.Tensor) -> torch.Tensor:
        """Reshape flattened model output ``[B, H*D]`` to ``[B, H, D]``."""
        if pred_flat.shape[-1] != self._flattened_dim:
            raise ValueError(
                f"Expected flattened dim {self._flattened_dim} "
                f"(horizon={self.action_horizon} * step_dim={self.step_action_dim}), "
                f"but model output has dim {pred_flat.shape[-1]}"
            )
        return pred_flat.view(-1, self.action_horizon, self.step_action_dim)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_in = self._prepare_batch(batch)
        pred_flat = self.model(model_in)
        pred_chunk = self._reshape_chunk(pred_flat)

        gt_chunk = batch["action_chunk"].to(self.device)

        loss = F.mse_loss(pred_chunk, gt_chunk)

        with torch.no_grad():
            mae = F.l1_loss(pred_chunk, gt_chunk)
            mae_first = F.l1_loss(pred_chunk[:, 0], gt_chunk[:, 0])

        return {
            "loss": loss,
            "mse": loss.detach(),
            "mae": mae.detach(),
            "mae_first": mae_first.detach(),
        }

    def predict_action(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        model_in = self._prepare_batch(batch)
        with torch.no_grad():
            pred_flat = self.model(model_in)
            pred_chunk = self._reshape_chunk(pred_flat)
        return {
            "action_chunk": pred_chunk.cpu(),
            "action": pred_chunk[:, 0].cpu(),  # first step for legacy compat
        }

    def supports_action_chunk(self) -> bool:
        return True


register_policy("action_chunk_bc", ActionChunkBCPolicy)
