"""Action head — continuous MLP regression for action prediction.

Architecture:
    Linear(input_dim, hidden_dim) → ReLU → Linear(hidden_dim, action_dim)

Input:  fused_feat Tensor[B, input_dim]  (default 128)
Output: action_pred Tensor[B, action_dim]  (default 2 for [dx, dy])
"""

from __future__ import annotations

import torch
from torch import nn


class ActionHead(nn.Module):
    """MLP action head for continuous action regression.

    Args:
        input_dim: Input fused feature dimension (default 128).
        hidden_dim: Hidden layer size (default 128).
        action_dim: Output action dimensionality (default 2).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """Predict action from fused features.

        Args:
            fused_feat: Tensor[B, input_dim].

        Returns:
            action_pred: Tensor[B, action_dim].
        """
        return self.net(fused_feat)


__all__ = ["ActionHead"]
