"""Small CNN vision encoder for Toy 2D images.

Architecture:
    Conv2d(3, 16, 3, padding=1) → ReLU → MaxPool2d(2)
    Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool2d(2)
    Conv2d(32, 64, 3, padding=1) → ReLU → AdaptiveAvgPool2d(1)
    Flatten → Linear(64, output_dim) → image_feat

Input:  Tensor[B, 3, H, W]   (H, W >= 16, typically 64)
Output: Tensor[B, output_dim] (default 128)
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNNVisionEncoder(nn.Module):
    """Lightweight CNN vision encoder suitable for Toy 2D 64×64 images.

    Args:
        output_dim: Output feature dimension (default 128).
    """

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image into a feature vector.

        Args:
            image: Tensor[B, 3, H, W] (H, W >= 16).

        Returns:
            image_feat: Tensor[B, output_dim].
        """
        conv_out = self.conv_layers(image)  # [B, 64, 1, 1]
        flattened = conv_out.view(conv_out.size(0), -1)  # [B, 64]
        image_feat = self.fc(flattened)  # [B, output_dim]
        return image_feat


__all__ = ["SmallCNNVisionEncoder"]
