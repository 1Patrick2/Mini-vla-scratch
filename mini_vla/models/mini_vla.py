"""MiniVLA — end-to-end vision-language-action model.

Assembles vision encoder, text backbone, state encoder, fusion, and action head
into a single forward pass.

Usage:
    model = MiniVLA(vision_encoder, text_encoder, state_encoder, fusion, action_head)
    action_pred = model(batch)  # Tensor[B, action_dim]
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class MiniVLA(nn.Module):
    """End-to-end Vision-Language-Action model.

    Args:
        vision_encoder: Module that maps ``image`` Tensor[B,3,H,W] → Tensor[B, D].
        text_encoder: Module that maps ``input_ids`` Tensor[B,T] + ``attention_mask`` → Tensor[B, D].
        state_encoder: Module that maps ``state`` Tensor[B, state_dim] → Tensor[B, D].
        fusion: Module that concatenates image/text/state features → Tensor[B, D].
        action_head: Module that maps fused features → Tensor[B, action_dim].
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        state_encoder: nn.Module,
        fusion: nn.Module,
        action_head: nn.Module,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.state_encoder = state_encoder
        self.fusion = fusion
        self.action_head = action_head

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run inference on a single batch.

        Args:
            batch: Dict containing ``image``, ``input_ids``, ``attention_mask``,
                   and ``state``.  The ``action`` key (if present) is ignored
                   during forward.

        Returns:
            action_pred: Tensor[B, action_dim].
        """
        image_feat = self.vision_encoder(batch["image"])
        text_feat = self.text_encoder(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        state_feat = self.state_encoder(batch["state"])
        fused_feat = self.fusion(image_feat, text_feat, state_feat)
        action_pred = self.action_head(fused_feat)
        return action_pred


__all__ = ["MiniVLA"]
