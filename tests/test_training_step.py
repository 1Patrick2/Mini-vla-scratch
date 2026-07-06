"""Tests for loss, metrics, and optimizer modules."""

import torch

from mini_vla.models import build_model
from mini_vla.training.losses import mse_action_loss
from mini_vla.training.metrics import mse, l1
from mini_vla.training.optimizer import create_optimizer

# Minimal yet dimensionally consistent config for optimizer tests.
_OPTIMIZER_CFG = {
    "model": {
        "action_dim": 2,
        "vision_encoder": {"type": "small_cnn", "output_dim": 16},
        "text_encoder": {"type": "mock_llm", "output_dim": 16, "freeze": True},
        "state_encoder": {"output_dim": 16},
        "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
        "action_head": {"input_dim": 16},
    },
    "train": {"lr": 0.01},
}


class TestMseActionLoss:
    """Loss function shape and value correctness."""

    def test_loss_is_scalar(self):
        pred = torch.randn(4, 2)
        gt = torch.randn(4, 2)
        loss = mse_action_loss(pred, gt)
        assert loss.ndim == 0  # scalar

    def test_identical_tensors_give_zero_loss(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = mse_action_loss(x, x)
        assert loss.item() == 0.0

    def test_loss_is_positive_for_different_tensors(self):
        pred = torch.zeros(4, 2)
        gt = torch.ones(4, 2)
        loss = mse_action_loss(pred, gt)
        assert loss.item() > 0


class TestMetrics:
    """Evaluation metrics."""

    def test_mse_value(self):
        pred = torch.tensor([[1.0, 2.0]])
        gt = torch.tensor([[1.0, 2.0]])
        assert mse(pred, gt) == 0.0

    def test_l1_value(self):
        pred = torch.tensor([[1.0, 2.0]])
        gt = torch.tensor([[1.0, 2.0]])
        assert l1(pred, gt) == 0.0


class TestOptimizer:
    """Optimizer creation — respects frozen parameters."""

    def test_optimizer_created(self):
        cfg = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 16},
                "text_encoder": {"type": "mock_llm", "output_dim": 16, "freeze": True},
                "state_encoder": {"output_dim": 16},
                "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
                "action_head": {"input_dim": 16},
            },
            "train": {"lr": 0.01},
        }
        model = build_model(cfg)
        opt = create_optimizer(model, cfg)
        assert opt is not None

    def test_frozen_text_encoder_not_in_optimizer(self):
        cfg = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 16},
                "text_encoder": {"type": "mock_llm", "output_dim": 16, "freeze": True},
                "state_encoder": {"output_dim": 16},
                "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
                "action_head": {"input_dim": 16},
            },
            "train": {"lr": 0.01},
        }
        model = build_model(cfg)
        opt = create_optimizer(model, cfg)

        # Collect param IDs that the optimizer tracks
        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        text_param_ids = {id(p) for p in model.text_encoder.parameters()}

        # Frozen text encoder params should NOT be in optimizer
        overlap = opt_param_ids & text_param_ids
        assert len(overlap) == 0, (
            f"Expected 0 frozen text params in optimizer, got {len(overlap)}"
        )

    def test_end_to_end_training_step(self, tmp_path):
        """End-to-end: generate data → dataset → dataloader → forward → loss → backward."""
        from pathlib import Path

        import yaml
        from torch.utils.data import DataLoader

        from mini_vla.datasets import Toy2DDataset
        from mini_vla.datasets.collate import collate_toy_2d

        # 1. Generate toy data
        from scripts.generate_toy_data import generate_toy_data
        data_root = generate_toy_data(
            output_root=tmp_path, num_episodes=2, max_steps=4,
            image_size=64, seed=42,
        )
        # 2. Build model from config dict
        cfg = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 16},
                "text_encoder": {"type": "mock_llm", "output_dim": 16, "freeze": True},
                "state_encoder": {"output_dim": 16},
                "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
                "action_head": {"input_dim": 16},
            },
            "train": {"lr": 0.01},
        }
        model = build_model(cfg)

        # 3. Create DataLoader
        ds = Toy2DDataset(data_root)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_toy_2d)
        batch = next(iter(loader))

        # 4. Forward
        pred = model(batch)
        loss = mse_action_loss(pred, batch["action"])

        # 5. Verify loss is finite scalar
        assert loss.ndim == 0
        assert torch.isfinite(loss)

        # 6. Backward
        loss.backward()

        # 7. Verify gradients flow to trainable params
        for name, param in model.action_head.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all()

        # 8. Frozen text_encoder should have no gradient
        for name, param in model.text_encoder.named_parameters():
            assert param.grad is None, f"{name} should not have gradient (frozen)"

        # 9. Optimizer step
        opt = create_optimizer(model, cfg)
        opt.step()
        opt.zero_grad()
