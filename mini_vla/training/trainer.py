"""Training orchestration for Mini VLA.

Provides a Trainer class that orchestrates data loading, model forward,
backward pass, and optimizer stepping for behavior cloning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from mini_vla.datasets.collate import collate_toy_2d
from mini_vla.datasets.factory import build_dataset
from mini_vla.models import build_model
from mini_vla.training.checkpoint import save_checkpoint
from mini_vla.training.losses import mse_action_loss
from mini_vla.training.metrics import l1
from mini_vla.training.optimizer import create_optimizer


class Trainer:
    """Behavior cloning trainer for MiniVLA.

    Args:
        config: Top-level merged config dictionary with ``model``, ``data``,
            and ``train`` sections.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device(config["train"].get("device", "cpu"))

        # Build model and move to device
        self.model = build_model(config).to(self.device)

        # Optimizer — only trainable (requires_grad=True) parameters
        self.optimizer = create_optimizer(self.model, config)

        # Checkpoint directory — use train.output_dir if specified
        train_cfg = config.get("train", {})
        custom_dir = train_cfg.get("output_dir")
        if custom_dir:
            self.checkpoint_dir = Path(custom_dir)
        else:
            self.checkpoint_dir = (
                Path(config.get("paths", {}).get("output_root", "outputs"))
                / "checkpoints"
            )

        # DataLoader
        self.train_loader = self._build_dataloader()

    def _build_dataloader(self) -> DataLoader:
        data_cfg = self.config["data"]
        ds = build_dataset(data_cfg)
        return DataLoader(
            ds,
            batch_size=data_cfg["batch_size"],
            shuffle=True,
            collate_fn=collate_toy_2d,
            num_workers=data_cfg.get("num_workers", 0),
        )

    def describe(self) -> str:
        """Return a human-readable description of the trainer configuration."""
        model_name = self.config["model"]["name"]
        dataset_type = self.config["data"]["dataset_type"]
        epochs = self.config["train"]["epochs"]
        return f"Trainer(model={model_name}, data={dataset_type}, epochs={epochs})"

    def train_one_epoch(self) -> Dict[str, float]:
        """Run one training epoch over the dataset.

        Returns:
            Dictionary with average ``loss`` and ``mae`` for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move tensors to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward
            action_pred = self.model(batch)
            loss = mse_action_loss(action_pred, batch["action"])

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_mae += l1(action_pred, batch["action"])
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "mae": total_mae / num_batches,
        }

    def fit(self, epochs: Optional[int] = None) -> None:
        """Run the full training loop.

        Args:
            epochs: Number of epochs to train.  Falls back to
                ``config["train"]["epochs"]`` if not provided.
        """
        if epochs is None:
            epochs = self.config["train"]["epochs"]

        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            metrics = self.train_one_epoch()
            print(
                f"Epoch {epoch}/{epochs}  "
                f"loss={metrics['loss']:.6f}  "
                f"mae={metrics['mae']:.6f}"
            )

            # Save last checkpoint
            save_checkpoint(
                self.checkpoint_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
            )

            # Save best checkpoint when loss improves
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=self.config,
                )


__all__ = ["Trainer"]
