"""Checkpoint save and load for MiniVLA training.

Saves model weights, optimizer state, epoch, metrics, and config to a single
``.pt`` file for resumable training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model, optimizer, and metadata to ``path``.

    Args:
        path: Output path (e.g. ``"outputs/checkpoints/last.pt"``).
        model: PyTorch model whose ``state_dict`` to save.
        optimizer: Optional optimizer whose ``state_dict`` to save.
        epoch: Current epoch number.
        metrics: Optional dictionary of scalar metrics (loss, mae, ...).
        config: Optional full configuration dict for reproducibility.

    Returns:
        The resolved ``path`` as a ``Path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    if config is not None:
        payload["config"] = config

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint from ``path`` and restore model (and optionally optimizer) state.

    Args:
        path: Checkpoint path.
        model: Model to load weights into.
        optimizer: Optional optimizer whose state to restore.
        device: Device to map the loaded state to.

    Returns:
        The full checkpoint payload dictionary containing ``model_state_dict``,
        ``epoch``, ``metrics``, and ``config`` keys.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload


__all__ = ["save_checkpoint", "load_checkpoint"]
