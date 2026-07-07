"""Tests for the evaluate_pusht CLI via subprocess."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter
from mini_vla.evaluation.evaluator import evaluate_policy_on_dataset
from mini_vla.inference.predictor import Predictor

RUN_REAL_PUSHT = os.getenv("RUN_REAL_PUSHT") == "1"
HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None


def _train_and_save_checkpoint(tmp_path):
    """Train a config-compatible MiniVLA and save a checkpoint."""
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
    # Use 128-dim config to match base.yaml
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
    return cfg, ckpt_path


class TestEvaluatePushTCLI:
    """CLI subprocess integration tests."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        _, self.ckpt_path = _train_and_save_checkpoint(tmp_path)
        self.report_path = tmp_path / "report.json"
        self.config_path = Path("configs/train/pusht_debug.yaml")

    def _run(self, *extra_args):
        cmd = [
            sys.executable,
            "scripts/evaluate_pusht.py",
            "--config", str(self.config_path),
            "--ckpt", str(self.ckpt_path),
            "--max-samples", "4",
            "--output", str(self.report_path),
            *extra_args,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    @pytest.mark.realdata
    @pytest.mark.skipif(
        not RUN_REAL_PUSHT or not HAS_LEROBOT,
        reason="Set RUN_REAL_PUSHT=1 and install lerobot to run real PushT vision test.",
    )
    def test_cli_with_remote_pusht(self):
        """CLI runs with real remote PushT (requires lerobot + network)."""
        result = self._run(
            "--repo-id", "lerobot/pusht",
            "--loader", "lerobot",
            "--max-samples", "8",
            "--heldout-ratio", "0.25",
            "--no-clip-action",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.report_path.exists()
        report = json.loads(self.report_path.read_text())
        assert report["dataset"]["mode"] == "vision"
        assert report["dataset"]["has_image"] is True
        assert "model" in report
        assert "baselines" in report

    def test_cli_with_mock_data(self):
        """CLI generates report.json via subprocess using --mock-data."""
        result = self._run("--mock-data")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.report_path.exists(), "Report JSON should exist"
        import json
        report = json.loads(self.report_path.read_text())
        assert "model" in report
        assert "baselines" in report
        assert "MAE" in result.stdout
        assert "Report saved" in result.stdout

    def test_cli_with_heldout_split(self):
        """CLI with --heldout-ratio produces a report with held-out semantics."""
        # Use 8 samples + heldout 0.5: 2 episodes, 1 held out
        result = self._run("--mock-data", "--heldout-ratio", "0.5", "--max-samples", "8")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.report_path.exists()
        report = json.loads(self.report_path.read_text())
        assert report["num_samples"] > 0
        # With mock data: 8 samples, 2 episodes, heldout=0.5
        # 1 episode (4 samples) held out for eval, 1 for train/baseline
        assert report["num_samples"] == 4, (
            f"Expected 4 held-out samples, got {report['num_samples']}"
        )

    def test_cli_mock_data_checks_baseline_source(self):
        """Mock CLI with heldout reports mean_action_source."""
        result = self._run("--mock-data", "--heldout-ratio", "0.5", "--max-samples", "8")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        report = json.loads(self.report_path.read_text())
        assert "mean_action_source" in report

    def test_evaluate_on_mock_dataset(self, tmp_path):
        """evaluate_policy_on_dataset works with mock PushT samples."""
        raw_samples = [
            {
                "observation.image": np.zeros((96, 96, 3), dtype=np.uint8),
                "observation.state": np.zeros(2, dtype=np.float32),
                "action": torch.tensor([float(i), float(i)]),
            }
            for i in range(4)
        ]
        # Convert to MiniVLA format via adapter
        adapter = PushTDatasetAdapter(raw_samples)
        samples = [adapter[i] for i in range(len(adapter))]

        cfg, ckpt_path = _train_and_save_checkpoint(tmp_path)
        predictor = Predictor(cfg, ckpt_path, clip_action=False)
        report = evaluate_policy_on_dataset(predictor, samples, action_dim=2)
        assert report["num_samples"] == 4
        assert "model" in report
        assert "baselines" in report

        out = tmp_path / "test_report.json"
        out.write_text(json.dumps(report, indent=2))
        assert out.exists()
