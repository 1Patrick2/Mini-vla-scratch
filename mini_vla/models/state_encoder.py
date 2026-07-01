"""State encoder — MLP mapping from robot state to feature vector.

Architecture:
    Linear(input_dim, hidden_dim) → ReLU → Linear(hidden_dim, output_dim)

Input:  Tensor[B, input_dim]   (default 2 for Toy 2D [x, y])
Output: Tensor[B, output_dim]  (default 128)
"""

from __future__ import annotations

import torch
from torch import nn


class StateEncoder(nn.Module):
    """MLP encoder for robot state.

    Args:
        input_dim: Dimensionality of the input state (default 2).
        hidden_dim: Hidden layer size (default 64).
        output_dim: Output feature dimension (default 128).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state into a feature vector.

        Args:
            state: Tensor[B, input_dim].

        Returns:
            state_feat: Tensor[B, output_dim].
        """
        return self.net(state)


__all__ = ["StateEncoder"]
