"""Tests for the infer_one CLI via subprocess."""

import subprocess
import sys
from pathlib import Path

import pytest


def _train_mini_checkpoint(tmp_path):
    """Train a tiny MiniVLA checkpoint for CLI testing."""
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
    # Use dimensions consistent with configs/base.yaml (output_dim=128)
    # so that configs/train/debug.yaml can be used directly in CLI.
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
        tmp_path / "model.pt", model,
        epoch=1, metrics={"loss": float(loss.item())},
    )
    return cfg, ckpt_path, data_root


class TestInferOneCLI:
    """CLI subprocess integration tests."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        _, self.ckpt_path, self.data_root = _train_mini_checkpoint(tmp_path)
        self.output_path = tmp_path / "pred.png"
        self.config_path = Path("configs/train/debug.yaml")

    def _run(self, *extra_args):
        cmd = [
            sys.executable,
            "scripts/infer_one.py",
            "--config", str(self.config_path),
            "--ckpt", str(self.ckpt_path),
            "--data-root", str(self.data_root),
            "--sample-index", "0",
            "--output", str(self.output_path),
            *extra_args,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    def test_end_to_end(self):
        """CLI runs successfully and produces expected output."""
        result = self._run()
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists(), "Output PNG should exist"
        assert "Pred action" in result.stdout
        assert "GT action" in result.stdout
        assert "L1 error" in result.stdout
        assert "Output" in result.stdout
        assert "Visualization saved" in result.stdout

    def test_with_no_clip_action(self):
        """--no-clip-action flag is accepted."""
        result = self._run("--no-clip-action")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists()

    def test_with_custom_arrow_scale(self):
        """--arrow-scale is accepted."""
        result = self._run("--arrow-scale", "2.0")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists()

    def test_missing_checkpoint_exits_with_error(self):
        """Missing checkpoint produces a clear error."""
        result = self._run("--ckpt", "nonexistent.pt")
        assert result.returncode != 0
        assert "checkpoint not found" in result.stderr.lower()

    def test_missing_data_root_exits_with_error(self):
        """Missing data root produces a clear error."""
        result = self._run("--data-root", "/nonexistent/path")
        assert result.returncode != 0
        assert "data root not found" in result.stderr.lower()

    def test_out_of_range_sample_index_exits_with_error(self):
        """Out-of-range sample index produces a clear error."""
        result = self._run("--sample-index", "9999")
        assert result.returncode != 0
        assert "out of range" in result.stderr.lower()
