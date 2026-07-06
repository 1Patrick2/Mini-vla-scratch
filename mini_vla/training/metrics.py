"""Training metrics for behavior cloning evaluation."""

from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error between predictions and targets."""
    return float(torch.nn.functional.mse_loss(pred, target).item())


def l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error (L1) between predictions and targets."""
    return float(torch.nn.functional.l1_loss(pred, target).item())


__all__ = [
    "mse",
    "l1",
]
