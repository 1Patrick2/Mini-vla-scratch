"""Tests for the Predictor / Policy inference."""

import pytest
import torch

from mini_vla.inference.predictor import Predictor


def _train_and_save_checkpoint(tmp_path):
    """Helper: train a MiniVLA for 1 epoch and save a checkpoint.

    Returns (config, checkpoint_path) for building a Predictor.
    """
    from torch.utils.data import DataLoader

    from mini_vla.datasets import Toy2DDataset
    from mini_vla.datasets.collate import collate_toy_2d
    from mini_vla.models import build_model
    from mini_vla.training.checkpoint import save_checkpoint
    from mini_vla.training.losses import mse_action_loss
    from mini_vla.training.optimizer import create_optimizer
    from scripts.generate_toy_data import generate_toy_data

    data_root = generate_toy_data(
        output_root=tmp_path / "data",
        num_episodes=2, max_steps=4, image_size=64, seed=42,
    )
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
    loader = DataLoader(
        Toy2DDataset(data_root), batch_size=4, collate_fn=collate_toy_2d,
    )
    opt = create_optimizer(model, cfg)
    for batch in loader:
        loss = mse_action_loss(model(batch), batch["action"])
        opt.zero_grad()
        loss.backward()
        opt.step()

    ckpt_path = save_checkpoint(
        tmp_path / "model.pt", model, optimizer=opt,
        epoch=1, metrics={"loss": float(loss.item())},
    )
    return cfg, ckpt_path, data_root


class TestPredictorConstruction:
    """Predictor can be built from config + checkpoint."""

    def test_predictor_constructs(self, tmp_path):
        cfg, ckpt_path, _ = _train_and_save_checkpoint(tmp_path)
        predictor = Predictor(cfg, ckpt_path)
        assert predictor.model is not None
        assert predictor.model.training is False  # eval mode


class TestPredictorOutput:
    """Predictor produces correct action tensors."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        cfg, ckpt_path, data_root = _train_and_save_checkpoint(tmp_path)
        from mini_vla.datasets import Toy2DDataset
        self.predictor = Predictor(cfg, ckpt_path)
        self.ds = Toy2DDataset(data_root)
        self.sample = self.ds[0]

    def test_predict(self):
        action = self.predictor.predict(self.sample)
        assert action.shape == (2,)

    def test_select_action_output_shape(self):
        action = self.predictor.select_action(self.sample)
        assert action.shape == (2,)

    def test_output_is_finite(self):
        action = self.predictor.predict(self.sample)
        assert torch.isfinite(action).all()

    def test_predict_without_action_key(self):
        """Predict still works when sample has no 'action' key."""
        no_action = {k: v for k, v in self.sample.items() if k != "action"}
        action = self.predictor.predict(no_action)
        assert action.shape == (2,)

    def test_default_clip_keeps_output_in_range(self):
        """Default Predictor with clipping keeps action within limit."""
        action = self.predictor.predict(self.sample)
        limit = self.predictor.action_limit  # default 0.05
        assert action.shape == (2,)
        assert torch.all(action.abs() <= limit + 1e-6), (
            f"Action {action} exceeds limit {limit}"
        )

    def test_clip_with_tiny_limit(self, tmp_path):
        """Tiny action_limit forces clipping — verify output respects bound."""
        cfg, ckpt_path, _ = _train_and_save_checkpoint(tmp_path)
        predictor = Predictor(
            cfg, ckpt_path,
            clip_action_flag=True, action_limit=1e-6,
        )
        action = predictor.predict(self.sample)
        assert action.shape == (2,)
        assert torch.all(action.abs() <= 1e-6 + 1e-8), (
            f"Clipped action {action} exceeds limit 1e-6"
        )

    def test_no_clip_returns_raw_action(self, tmp_path):
        """With clip_action_flag=False, action can exceed the default limit."""
        cfg, ckpt_path, _ = _train_and_save_checkpoint(tmp_path)
        predictor = Predictor(
            cfg, ckpt_path,
            clip_action_flag=False, action_limit=0.05,
        )
        action = predictor.predict(self.sample)
        assert action.shape == (2,)
