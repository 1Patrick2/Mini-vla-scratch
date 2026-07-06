"""Checkpoint save and load tests."""

from pathlib import Path

import pytest
import torch

from mini_vla.training.checkpoint import load_checkpoint, save_checkpoint


def _make_dummy_model():
    """Return a simple linear model for testing."""
    return torch.nn.Linear(4, 2)


class TestSaveCheckpoint:
    def test_save_creates_file(self, tmp_path):
        model = _make_dummy_model()
        path = save_checkpoint(tmp_path / "test.pt", model)
        assert path.exists()

    def test_save_with_optimizer(self, tmp_path):
        model = _make_dummy_model()
        opt = torch.optim.Adam(model.parameters())
        path = save_checkpoint(tmp_path / "opt.pt", model, optimizer=opt)
        assert path.exists()
        payload = torch.load(path, weights_only=True)
        assert "optimizer_state_dict" in payload

    def test_save_with_epoch_and_metrics(self, tmp_path):
        model = _make_dummy_model()
        metrics = {"loss": 0.042, "mae": 0.11}
        path = save_checkpoint(
            tmp_path / "meta.pt", model, epoch=5, metrics=metrics
        )
        payload = torch.load(path, weights_only=True)
        assert payload["epoch"] == 5
        assert payload["metrics"] == metrics


class TestLoadCheckpoint:
    def test_load_restores_weights(self, tmp_path):
        model = _make_dummy_model()
        with torch.no_grad():
            model.weight.fill_(3.0)
        save_checkpoint(tmp_path / "w.pt", model)

        new_model = _make_dummy_model()
        load_checkpoint(tmp_path / "w.pt", new_model)
        assert torch.allclose(
            new_model.weight, torch.full_like(new_model.weight, 3.0)
        )

    def test_load_restores_epoch_and_metrics(self, tmp_path):
        model = _make_dummy_model()
        save_checkpoint(tmp_path / "m.pt", model, epoch=3, metrics={"loss": 0.5})

        new_model = _make_dummy_model()
        payload = load_checkpoint(tmp_path / "m.pt", new_model)
        assert payload["epoch"] == 3
        assert payload["metrics"]["loss"] == 0.5

    def test_load_restores_optimizer(self, tmp_path):
        model = _make_dummy_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        save_checkpoint(tmp_path / "o.pt", model, optimizer=opt)

        new_model = _make_dummy_model()
        new_opt = torch.optim.SGD(new_model.parameters(), lr=0.1)
        load_checkpoint(tmp_path / "o.pt", new_model, optimizer=new_opt)
        assert new_opt.state_dict()["param_groups"][0]["lr"] == 0.1

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pt", _make_dummy_model())

    def test_model_can_forward_after_load(self, tmp_path):
        model = _make_dummy_model()
        save_checkpoint(tmp_path / "fwd.pt", model)

        new_model = _make_dummy_model()
        load_checkpoint(tmp_path / "fwd.pt", new_model)
        x = torch.randn(2, 4)
        out = new_model(x)
        assert out.shape == (2, 2)


class TestCheckpointIntegration:
    """Checkpointing integrated with the Trainer."""

    def test_trainer_saves_checkpoint_after_fit(self, tmp_path):
        from scripts.generate_toy_data import generate_toy_data
        data_root = generate_toy_data(
            output_root=tmp_path / "data",
            num_episodes=2, max_steps=4, image_size=64, seed=42,
        )
        config = {
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
            "train": {"epochs": 2, "lr": 0.001, "device": "cpu"},
            "paths": {"output_root": str(tmp_path / "outputs")},
        }
        from mini_vla.training.trainer import Trainer
        trainer = Trainer(config)
        trainer.fit()

        last = tmp_path / "outputs" / "checkpoints" / "last.pt"
        best = tmp_path / "outputs" / "checkpoints" / "best.pt"
        assert last.exists(), "last.pt checkpoint should exist"
        assert best.exists(), "best.pt checkpoint should exist"

    def test_checkpoint_contains_expected_keys(self, tmp_path):
        model = _make_dummy_model()
        opt = torch.optim.Adam(model.parameters())
        metrics = {"loss": 0.1, "mae": 0.2}
        path = save_checkpoint(
            tmp_path / "ckpt.pt", model, optimizer=opt,
            epoch=3, metrics=metrics, config={"model": {"name": "mini_vla"}},
        )
        payload = torch.load(path, weights_only=True)
        assert "model_state_dict" in payload
        assert "optimizer_state_dict" in payload
        assert payload["epoch"] == 3
        assert payload["metrics"]["loss"] == 0.1
        assert payload["config"]["model"]["name"] == "mini_vla"
