"""LLM-ready text encoder — base, mock, and HuggingFace adapter.

Architecture:
    BaseTextEncoder (abstract)
    ├── MockLLMTextEncoder: Embedding + mask-aware pooling + projection
    └── LLMTextEncoder: HuggingFace AutoModel adapter (optional)

Input:  input_ids Tensor[B, T], attention_mask Tensor[B, T] | None
Output: text_feat Tensor[B, output_dim] (default 128)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class BaseTextEncoder(nn.Module):
    """Abstract base for all text encoders."""

    output_dim: int

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode tokenised instruction into a fixed-size feature vector.

        Args:
            input_ids: Tensor[B, T] — token IDs.
            attention_mask: Tensor[B, T] or None — 1 for real tokens, 0 for pad.

        Returns:
            text_feat: Tensor[B, output_dim].
        """
        raise NotImplementedError


class MockLLMTextEncoder(BaseTextEncoder):
    """Mock text encoder for testing — no ``transformers`` dependency.

    Uses Embedding + mask-aware mean pooling + Linear projection.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)  # [B, T, H]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = emb.mean(dim=1)  # [B, H]

        text_feat = self.projection(pooled)  # [B, output_dim]
        return text_feat


class LLMTextEncoder(BaseTextEncoder):
    """HuggingFace / local LLM text encoder (optional dependency).

    Only usable when ``transformers`` is installed.
    Raises a clear ``ImportError`` if ``transformers`` is not available.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dim: int = 128,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.freeze = freeze

        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is required for LLMTextEncoder, "
                "but not required for MockLLMTextEncoder."
            ) from None

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden_dim, output_dim)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use last hidden state mean pooling
        hidden = outputs.last_hidden_state  # [B, T, H]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)
        text_feat = self.projection(pooled)
        return text_feat


__all__ = [
    "BaseTextEncoder",
    "MockLLMTextEncoder",
    "LLMTextEncoder",
]
