"""SingleFrame BC policy — predict action from current observation.

Input:  image + instruction + state
Output: action (single step)
Target: absolute action regression
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from mini_vla.policies.base import BasePolicy
from mini_vla.policies.registry import register_policy
from mini_vla.training.losses import mse_action_loss


class SingleFrameBCPolicy(BasePolicy):
    """Single-frame Behavior Cloning policy.

    Predicts action directly from current observation (image, instruction,
    state) using the underlying MiniVLA model.
    """

    policy_type = "single_frame_bc"

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


register_policy("single_frame_bc", SingleFrameBCPolicy)
