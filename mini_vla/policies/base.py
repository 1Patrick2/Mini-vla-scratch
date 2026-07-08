"""Base policy class for MiniVLA.

All policies inherit from ``BasePolicy``, which is itself a ``nn.Module``.
The base class provides common machinery: MiniVLA model building, device
management, and the abstract interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn

from mini_vla.models import build_model


class BasePolicy(nn.Module, ABC):
    """Abstract base class for all MiniVLA policies.

    Every policy owns an underlying MiniVLA model and exposes:

    * ``compute_loss(batch) -> dict``  — used by ``Trainer``
    * ``predict_action(batch) -> dict`` — used by ``Evaluator`` / rollout
    """

    policy_type: str = "base"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        # Build the underlying MiniVLA model
        self.model = build_model(config)
        self._config = config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ── Abstract interface ──────────────────────────────────────────────

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training loss and auxiliary metrics.

        Args:
            batch: A collated batch of samples with at least ``action``.

        Returns:
            Dictionary with ``loss``, ``mse``, ``mae`` (all detached tensors).
        """
        ...

    @abstractmethod
    def predict_action(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run inference and return the predicted action(s).

        Args:
            batch: A collated batch of samples.

        Returns:
            Dictionary with at least ``action`` (Tensor).
        """
        ...

    def get_action_dim(self) -> int:
        """Return the action dimension that this policy predicts."""
        return self._config.get("model", self._config).get("action_dim", 2)

    def supports_action_chunk(self) -> bool:
        return False

    # ── Convenience forward (used internally) ───────────────────────────

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Move relevant tensors to the policy's device."""
        keys = {"image", "input_ids", "attention_mask", "state"}
        return {
            k: v.to(self.device) if k in keys and isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k in keys
        }


__all__ = ["BasePolicy"]
