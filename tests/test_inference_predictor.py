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
            clip_action=True, action_limit=1e-6,
        )
        action = predictor.predict(self.sample)
        assert action.shape == (2,)
        assert torch.all(action.abs() <= 1e-6 + 1e-8), (
            f"Clipped action {action} exceeds limit 1e-6"
        )

    def test_predictor_matches_manual_loaded_model(self, tmp_path):
        """Predictor raw output (clip_action=False) matches manual forward."""

        from mini_vla.datasets import Toy2DDataset
        from mini_vla.models import build_model
        from mini_vla.training.checkpoint import load_checkpoint

        # Fresh training run with dedicated data
        data_root = _train_and_save_checkpoint(tmp_path / "train")[2]
        cfg, ckpt_path, _ = _train_and_save_checkpoint(tmp_path)

        # Predictor with no clipping
        predictor = Predictor(cfg, ckpt_path, clip_action=False)
        sample = Toy2DDataset(data_root)[0]
        pred = predictor.predict(sample)

        # Manual forward
        model = build_model(cfg)
        load_checkpoint(ckpt_path, model)
        model.eval()
        batch = {
            "image": sample["image"].unsqueeze(0),
            "input_ids": sample["input_ids"].unsqueeze(0),
            "attention_mask": sample["attention_mask"].unsqueeze(0),
            "state": sample["state"].unsqueeze(0),
        }
        with torch.no_grad():
            manual_pred = model(batch).squeeze(0).cpu()

        assert torch.allclose(pred, manual_pred, atol=1e-6), (
            f"Predictor output {pred} differs from manual forward {manual_pred}"
        )


class TestInferenceQuality:
    """Inference quality: trained Predictor should outperform zero-action baseline."""

    def _train_full_model(self, tmp_path):
        """Train a config-compatible MiniVLA for enough epochs and return (predictor, dataset)."""
        from mini_vla.datasets import Toy2DDataset
        from scripts.generate_toy_data import generate_toy_data

        data_root = generate_toy_data(
            output_root=tmp_path / "data",
            num_episodes=4, max_steps=8, image_size=64, seed=42,
        )
        # Use 128-dim config compatible with configs/base.yaml
        cfg = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 128},
                "text_encoder": {"type": "mock_llm", "output_dim": 128, "freeze": True},
                "state_encoder": {"input_dim": 2, "output_dim": 128},
                "fusion": {"type": "concat_mlp", "input_dim": 384, "output_dim": 128},
                "action_head": {"input_dim": 128},
            },
            "train": {"lr": 0.01},
        }
        from torch.utils.data import DataLoader

        from mini_vla.datasets.collate import collate_toy_2d
        from mini_vla.models import build_model
        from mini_vla.training.checkpoint import save_checkpoint
        from mini_vla.training.losses import mse_action_loss
        from mini_vla.training.optimizer import create_optimizer

        model = build_model(cfg)
        loader = DataLoader(
            Toy2DDataset(data_root), batch_size=4, collate_fn=collate_toy_2d,
        )
        opt = create_optimizer(model, cfg)
        for _ in range(50):
            for batch in loader:
                loss = mse_action_loss(model(batch), batch["action"])
                opt.zero_grad()
                loss.backward()
                opt.step()

        ckpt_path = save_checkpoint(
            tmp_path / "model.pt", model,
            epoch=30, metrics={"loss": float(loss.item())},
        )
        predictor = Predictor(cfg, ckpt_path, clip_action=False)
        ds = Toy2DDataset(data_root)
        return predictor, ds

    def test_trained_predictor_beats_zero_baseline(self, tmp_path):
        """Raw Predictor output should have lower L1 than zero-action baseline."""
        predictor, ds = self._train_full_model(tmp_path)

        num_samples = min(len(ds), 12)
        pred_l1_sum = 0.0
        zero_l1_sum = 0.0

        for i in range(num_samples):
            sample = ds[i]
            pred_action = predictor.predict(sample)
            gt_action = sample["action"]

            pred_l1_sum += float(torch.abs(pred_action - gt_action).mean())
            zero_l1_sum += float(torch.abs(torch.zeros_like(gt_action) - gt_action).mean())

        pred_l1 = pred_l1_sum / num_samples
        zero_l1 = zero_l1_sum / num_samples

        assert pred_l1 < zero_l1, (
            f"Predictor L1 ({pred_l1:.6f}) should be lower than "
            f"zero-action L1 ({zero_l1:.6f})"
        )
