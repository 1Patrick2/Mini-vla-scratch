"""Consolidated tests for the complete training pipeline.

Covers loss functions, metrics, optimizer, end-to-end training step,
trainer loop, checkpoint save/load, and training correctness.
"""

import pytest
import torch

from mini_vla.models import build_model
from mini_vla.training.checkpoint import load_checkpoint, save_checkpoint
from mini_vla.training.losses import mse_action_loss
from mini_vla.training.metrics import l1, mse
from mini_vla.training.optimizer import create_optimizer
from mini_vla.training.trainer import Trainer

# Minimal yet dimensionally consistent model config shared across tests.
_MODEL_CFG = {
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

_TRAINER_CFG = {
    **_MODEL_CFG,
    "data": {
        "dataset_type": "toy_2d",
        "batch_size": 4,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_dummy_model():
    """Return a simple linear model for testing."""
    return torch.nn.Linear(4, 2)


def _generate_toy_data(tmp_path, num_episodes=2, max_steps=4):
    """Generate toy 2D data and return the data root path."""
    from scripts.generate_toy_data import generate_toy_data
    return generate_toy_data(
        output_root=tmp_path,
        num_episodes=num_episodes,
        max_steps=max_steps,
        image_size=64,
        seed=42,
    )


def _build_data_loader(data_root, cfg=None):
    """Build a DataLoader from generated data."""
    from torch.utils.data import DataLoader

    from mini_vla.datasets import Toy2DDataset
    from mini_vla.datasets.collate import collate_toy_2d

    ds = Toy2DDataset(data_root)
    return DataLoader(
        ds,
        batch_size=4 if cfg is None else cfg["data"]["batch_size"],
        collate_fn=collate_toy_2d,
    )


def _evaluate_loss(model, loader):
    """Compute average MSE loss over the full loader.

    Args:
        model: MiniVLA model.
        loader: DataLoader yielding batches with ``action`` key.

    Returns:
        Scalar float — average MSE loss across all batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            total_loss += mse_action_loss(pred, batch["action"]).item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


def _zero_action_baseline_loss(loader):
    """Compute average MSE loss when predicting all-zero actions.

    Args:
        loader: DataLoader yielding batches with ``action`` key.

    Returns:
        Scalar float — average MSE loss of zero prediction.
    """
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        zero_pred = torch.zeros_like(batch["action"])
        total_loss += mse_action_loss(zero_pred, batch["action"]).item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


# ── Loss ──────────────────────────────────────────────────────────────────────

class TestMseActionLoss:
    """Loss function shape and value correctness."""

    def test_loss_is_scalar(self):
        pred = torch.randn(4, 2)
        gt = torch.randn(4, 2)
        loss = mse_action_loss(pred, gt)
        assert loss.ndim == 0

    def test_identical_tensors_give_zero_loss(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = mse_action_loss(x, x)
        assert loss.item() == 0.0

    def test_loss_is_positive_for_different_tensors(self):
        pred = torch.zeros(4, 2)
        gt = torch.ones(4, 2)
        loss = mse_action_loss(pred, gt)
        assert loss.item() > 0


# ── Metrics ───────────────────────────────────────────────────────────────────

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


# ── Optimizer ─────────────────────────────────────────────────────────────────

class TestOptimizer:
    """Optimizer creation — respects frozen parameters."""

    def test_optimizer_created(self):
        model = build_model(_MODEL_CFG)
        opt = create_optimizer(model, _MODEL_CFG)
        assert opt is not None

    def test_frozen_text_encoder_not_in_optimizer(self):
        model = build_model(_MODEL_CFG)
        opt = create_optimizer(model, _MODEL_CFG)

        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        text_param_ids = {id(p) for p in model.text_encoder.parameters()}

        overlap = opt_param_ids & text_param_ids
        assert len(overlap) == 0, (
            f"Expected 0 frozen text params in optimizer, got {len(overlap)}"
        )


# ── Trainer ───────────────────────────────────────────────────────────────────

class TestTrainerDescribe:
    """Backward-compatible describe() behavior."""

    def test_trainer_describes_loaded_config(self, tmp_path):
        data_root = _generate_toy_data(tmp_path)
        from mini_vla.config import load_config
        config = load_config("configs/train/debug.yaml")
        config["data"]["data_root"] = str(data_root)
        trainer = Trainer(config)
        assert trainer.describe() == (
            "Trainer(model=mini_vla, data=toy_2d, epochs=1)"
        )


class TestTrainerTraining:
    """Training loop behavior with real generated data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        data_root = _generate_toy_data(tmp_path)
        self.config = {
            **_TRAINER_CFG,
            "data": {**_TRAINER_CFG["data"], "data_root": str(data_root)},
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
        changed = any(not torch.equal(before[k], after[k]) for k in before)
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
        all_same = all(torch.equal(before[k], after[k]) for k in before)
        assert all_same, "Frozen text_encoder parameters should not change"


# ── Checkpoint ────────────────────────────────────────────────────────────────

class TestSaveCheckpoint:
    """Checkpoint save operations."""

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
    """Checkpoint load operations."""

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
    """Checkpointing integrated with the Trainer and MiniVLA."""

    def test_trainer_saves_checkpoint_after_fit(self, tmp_path):
        data_root = _generate_toy_data(tmp_path / "data")
        config = {
            **_TRAINER_CFG,
            "data": {**_TRAINER_CFG["data"], "data_root": str(data_root)},
            "train": {"epochs": 2, "lr": 0.001, "device": "cpu"},
            "paths": {"output_root": str(tmp_path / "outputs")},
        }
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

    def test_minivla_checkpoint_reload_consistency(self, tmp_path):
        """MiniVLA checkpoint reload should produce identical predictions."""
        data_root = _generate_toy_data(tmp_path / "data")

        # Build and train model for 1 epoch
        model = build_model(_MODEL_CFG)
        loader = _build_data_loader(data_root)
        opt = create_optimizer(model, _MODEL_CFG)
        for batch in loader:
            loss = mse_action_loss(model(batch), batch["action"])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Reference predictions before save
        ref_batch = next(iter(loader))
        ref_pred = model(ref_batch)

        # Save and reload
        ckpt_path = save_checkpoint(
            tmp_path / "ckpt.pt", model, optimizer=opt,
            epoch=1, metrics={"loss": float(loss.item())},
        )
        loaded_model = build_model(_MODEL_CFG)
        load_checkpoint(ckpt_path, loaded_model)

        # Must produce identical predictions
        loaded_pred = loaded_model(ref_batch)
        assert torch.allclose(ref_pred, loaded_pred, atol=1e-6), (
            "Checkpoint reload produced different predictions"
        )


# ── End-to-end ────────────────────────────────────────────────────────────────

class TestEndToEndTrainingStep:
    """Single training step through the full pipeline."""

    def test_full_step_from_dataloader(self, tmp_path):
        """Generate data → DataLoader → forward → loss → backward → step."""
        data_root = _generate_toy_data(tmp_path)
        model = build_model(_MODEL_CFG)
        loader = _build_data_loader(data_root)
        batch = next(iter(loader))

        # Forward
        pred = model(batch)
        loss = mse_action_loss(pred, batch["action"])

        assert loss.ndim == 0
        assert torch.isfinite(loss)

        # Backward
        loss.backward()

        # Gradients: trainable params have grad; frozen text_encoder has none
        for name, param in model.action_head.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all()

        for name, param in model.text_encoder.named_parameters():
            assert param.grad is None, f"{name} should have no gradient (frozen)"

        # Optimizer step
        opt = create_optimizer(model, _MODEL_CFG)
        opt.step()
        opt.zero_grad()


# ── Correctness ───────────────────────────────────────────────────────────────

class TestTrainingCorrectness:
    """Training correctness: model should learn better than baselines."""

    def test_trained_model_reduces_loss(self, tmp_path):
        """Training should reduce loss below initial model loss."""
        data_root = _generate_toy_data(tmp_path, num_episodes=4, max_steps=8)
        model = build_model(_MODEL_CFG)
        loader = _build_data_loader(data_root)

        initial_loss = _evaluate_loss(model, loader)
        zero_loss = _zero_action_baseline_loss(loader)

        opt = create_optimizer(model, _MODEL_CFG)
        num_epochs = 50
        for _ in range(num_epochs):
            for batch in loader:
                pred = model(batch)
                loss = mse_action_loss(pred, batch["action"])
                opt.zero_grad()
                loss.backward()
                opt.step()

        final_loss = _evaluate_loss(model, loader)

        assert final_loss < initial_loss, (
            f"Final loss {final_loss:.6f} should be lower than "
            f"initial loss {initial_loss:.6f} "
            f"(zero baseline: {zero_loss:.6f})"
        )

    def test_tiny_dataset_overfit(self, tmp_path):
        """Model should overfit a tiny dataset, showing loss trending down."""
        data_root = _generate_toy_data(tmp_path, num_episodes=4, max_steps=8)
        model = build_model(_MODEL_CFG)
        loader = _build_data_loader(data_root)

        initial_loss = _evaluate_loss(model, loader)

        opt = create_optimizer(model, _MODEL_CFG)
        num_epochs = 20
        for _ in range(num_epochs):
            for batch in loader:
                pred = model(batch)
                loss = mse_action_loss(pred, batch["action"])
                opt.zero_grad()
                loss.backward()
                opt.step()

        final_loss = _evaluate_loss(model, loader)

        assert final_loss < initial_loss, (
            f"Final loss {final_loss:.6f} should be lower than "
            f"initial loss {initial_loss:.6f}"
        )
