"""DeltaAction BC policy — predict residual action (delta).

Input:  image + instruction + state + prev_state + prev_action
Output: delta_action (residual = action_t - prev_action_t)
Target: MSE on delta_action; reconstructed via prev_action + pred_delta
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from mini_vla.policies.base import BasePolicy
from mini_vla.policies.registry import register_policy
from mini_vla.training.losses import mse_action_loss


class DeltaActionBCPolicy(BasePolicy):
    """Delta/Residual Action Behavior Cloning policy.

    Predicts the residual ``action_t - prev_action_t`` and reconstructs
    the absolute action via ``prev_action + pred_delta`` during inference.
    """

    policy_type = "delta_action_bc"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_in = self._prepare_batch(batch)
        pred_delta = self.model(model_in)
        gt = batch["action"].to(self.device)  # target is already delta in dataset

        loss = mse_action_loss(pred_delta, gt)

        with torch.no_grad():
            mae_val = torch.nn.functional.l1_loss(pred_delta, gt)

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
            pred_delta = self.model(model_in)

        # Reconstruct absolute action from delta
        prev_action = batch.get("prev_action", batch.get("prev_action_normalized"))
        if prev_action is not None:
            action = prev_action.to(self.device) + pred_delta
        else:
            # Fallback: just return delta (should not happen in well-formed eval)
            action = pred_delta

        return {"action": action.cpu()}


register_policy("delta_action_bc", DeltaActionBCPolicy)
