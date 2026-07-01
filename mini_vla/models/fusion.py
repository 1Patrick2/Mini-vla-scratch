"""Multimodal fusion — concatenates image, text, and state features.

Architecture:
    Concat(image_feat, text_feat, state_feat) → Linear → ReLU → Linear → ReLU

Input:  image_feat Tensor[B, D], text_feat Tensor[B, D], state_feat Tensor[B, D]
Output: fused_feat Tensor[B, output_dim]  (default 128)
"""

from __future__ import annotations

import torch
from torch import nn


class FusionMLP(nn.Module):
    """Fuses vision, language, and state features via concatenation + MLP.

    Args:
        input_dim: Sum of all feature dimensions (default 384 for 3×128).
        hidden_dim: Hidden layer size (default 256).
        output_dim: Output fused feature dimension (default 128).
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        image_feat: torch.Tensor,
        text_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal features.

        Args:
            image_feat: Tensor[B, D].
            text_feat: Tensor[B, D].
            state_feat: Tensor[B, D].

        Returns:
            fused_feat: Tensor[B, output_dim].
        """
        fused = torch.cat([image_feat, text_feat, state_feat], dim=-1)
        return self.net(fused)


__all__ = ["FusionMLP"]
