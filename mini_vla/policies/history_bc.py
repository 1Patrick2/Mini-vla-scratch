"""History BC policy — predict action with temporal context.

Input:  image + instruction + state + prev_state + prev_action
Output: action (single step)
Target: absolute action regression with temporal context

The underlying MiniVLA model receives the concatenated history state
constructed by the dataset wrapper, typically ``state + prev_state + prev_action``.
The exact input dimension is configured through ``shape_meta`` / ``model.state_dim``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from mini_vla.policies.base import BasePolicy
from mini_vla.policies.registry import register_policy
from mini_vla.training.losses import mse_action_loss


class HistoryBCPolicy(BasePolicy):
    """History-augmented Behavior Cloning policy.

    Receives previous state/action as additional input and predicts
    the current action.
    """

    policy_type = "history_bc"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_in = self._prepare_batch(batch)
        action_pred = self.model(model_in)
        gt = batch["action"].to(self.device)

        loss = mse_action_loss(action_pred, gt)

        with torch.no_grad():
            mae_val = torch.nn.functional.l1_loss(action_pred, gt)

        return {
            "loss": loss,
            "mse": loss.detach(),
            "mae": mae_val.detach(),
        }

    def predict_action(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        model_in = self._prepare_batch(batch)
        with torch.no_grad():
            action_pred = self.model(model_in)
        return {"action": action_pred.cpu()}


register_policy("history_bc", HistoryBCPolicy)
