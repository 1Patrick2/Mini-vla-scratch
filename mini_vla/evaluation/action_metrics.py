"""Action prediction metrics for offline evaluation."""

from __future__ import annotations

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error (L1)."""
    return float(torch.abs(pred - target).mean().item())


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error."""
    return float(torch.nn.functional.mse_loss(pred, target).item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error."""
    return float(torch.sqrt(torch.nn.functional.mse_loss(pred, target)).item())


def per_dim_mae(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    """Per-dimension MAE.

    Returns a list of length ``action_dim``.
    """
    return torch.abs(pred - target).mean(dim=0).tolist()


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Cosine similarity between predicted and target actions.

    Returns a value in ``[-1, 1]`` (higher is better).
    """
    pred_norm = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    target_norm = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return float((pred_norm * target_norm).sum(dim=-1).mean().item())


def finite_ratio(pred: torch.Tensor) -> float:
    """Ratio of finite (non-NaN, non-Inf) elements."""
    return float(torch.isfinite(pred).all(dim=-1).float().mean().item())


def clip_rate(
    raw_action: torch.Tensor,
    clipped_action: torch.Tensor,
    atol: float = 1e-8,
) -> float:
    """Fraction of samples where clipping changed the action."""
    return float(
        (torch.abs(raw_action - clipped_action) > atol).any(dim=-1).float().mean().item()
    )


__all__ = [
    "clip_rate",
    "cosine_similarity",
    "finite_ratio",
    "mae",
    "mse",
    "per_dim_mae",
    "rmse",
]
