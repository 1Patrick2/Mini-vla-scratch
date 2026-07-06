"""Optimizer factory for MiniVLA training.

Creates an Adam optimizer that only updates trainable parameters
(respecting frozen modules such as ``text_encoder``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from torch import nn, optim


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> optim.Optimizer:
    """Create an Adam optimizer for trainable model parameters only.

    Frozen parameters (``requires_grad=False``) are excluded.

    Args:
        model: The MiniVLA model.
        config: Training config dict with ``lr`` key (default ``0.001``).

    Returns:
        ``Adam`` optimizer targeting only trainable parameters.
    """
    train_cfg = config.get("train", config)
    lr = float(train_cfg.get("lr", 0.001))

    params: Iterable[nn.Parameter] = (
        p for p in model.parameters() if p.requires_grad
    )
    return optim.Adam(params, lr=lr)


__all__ = [
    "create_optimizer",
]
