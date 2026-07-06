"""Training losses for behavior cloning.

Loss functions receive batched prediction and ground-truth tensors
and return scalar loss values.
"""

from __future__ import annotations

import torch
from torch import nn


def mse_action_loss(
    action_pred: torch.Tensor,
    action_gt: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between predicted and ground-truth actions.

    Args:
        action_pred: Tensor[B, action_dim] — model output.
        action_gt:   Tensor[B, action_dim] — expert action.

    Returns:
        A scalar ``Tensor`` suitable for ``backward()``.
    """
    return nn.functional.mse_loss(action_pred, action_gt)


__all__ = [
    "mse_action_loss",
]
