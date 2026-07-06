"""Tests for the Trainer class."""

import pytest
import torch

from mini_vla.training.trainer import Trainer


class TestTrainerDescribe:
    """Backward-compatible describe() behavior."""

    def test_trainer_describes_loaded_config(self, tmp_path):
        from scripts.generate_toy_data import generate_toy_data
        data_root = generate_toy_data(
            output_root=tmp_path,
            num_episodes=2, max_steps=4, image_size=64, seed=42,
        )
        from mini_vla.config import load_config
        config = load_config("configs/train/debug.yaml")
        # Point data_root to generated data
        config["data"]["data_root"] = str(data_root)
        trainer = Trainer(config)
        assert trainer.describe() == (
            "Trainer(model=mini_vla, data=toy_2d, epochs=1)"
        )


class TestTrainerTraining:
    """Training loop behavior with real generated data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from scripts.generate_toy_data import generate_toy_data
        data_root = generate_toy_data(
            output_root=tmp_path,
            num_episodes=2,
            max_steps=4,
            image_size=64,
            seed=42,
        )
        self.config = {
            "model": {
                "name": "mini_vla",
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 16},
                "text_encoder": {"type": "mock_llm", "output_dim": 16,
                                 "freeze": True},
                "state_encoder": {"output_dim": 16},
                "fusion": {"type": "concat_mlp", "input_dim": 48,
                           "output_dim": 16},
                "action_head": {"input_dim": 16},
            },
            "data": {
                "dataset_type": "toy_2d",
                "data_root": str(data_root),
                "batch_size": 4,
            },
            "train": {"epochs": 1, "lr": 0.001, "device": "cpu"},
        }

    def test_trainer_constructs(self):
        trainer = Trainer(self.config)
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None

    def test_train_one_epoch_returns_metrics(self):
        trainer = Trainer(self.config)
        metrics = trainer.train_one_epoch()
        assert "loss" in metrics
        assert "mae" in metrics
        assert metrics["loss"] >= 0.0
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_fit_runs_without_error(self):
        trainer = Trainer(self.config)
        trainer.fit(epochs=2)
        # No assertions — must not crash across 2 epochs

    def test_trainable_params_updated_after_fit(self):
        trainer = Trainer(self.config)
        before = {
            name: param.clone()
            for name, param in trainer.model.action_head.named_parameters()
        }
        trainer.fit(epochs=2)
        after = {
            name: param.clone()
            for name, param in trainer.model.action_head.named_parameters()
        }
        changed = any(
            not torch.equal(before[k], after[k]) for k in before
        )
        assert changed, "action_head parameters should update after training"

    def test_frozen_text_encoder_unchanged(self):
        trainer = Trainer(self.config)
        before = {
            name: param.clone()
            for name, param in trainer.model.text_encoder.named_parameters()
        }
        trainer.fit(epochs=2)
        after = {
            name: param.clone()
            for name, param in trainer.model.text_encoder.named_parameters()
        }
        all_same = all(
            torch.equal(before[k], after[k]) for k in before
        )
        assert all_same, "Frozen text_encoder parameters should not change"
