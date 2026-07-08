"""Tests for create_episode_split CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestCreateEpisodeSplitCLI:
    """Subprocess tests with --mock-data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.output_path = tmp_path / "split.json"

    def test_mock_pusht(self):
        result = subprocess.run([
            sys.executable, "scripts/create_episode_split.py",
            "--dataset-name", "pusht",
            "--mock-data",
            "--max-samples", "32",
            "--train-ratio", "0.8",
            "--seed", "42",
            "--output", str(self.output_path),
        ], capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists()
        data = json.loads(self.output_path.read_text())
        assert "train_episode_ids" in data
        assert "eval_episode_ids" in data
        assert len(data["train_episode_ids"]) > 0
        assert len(data["eval_episode_ids"]) > 0

    def test_mock_pusht_no_overlap(self):
        result = subprocess.run([
            sys.executable, "scripts/create_episode_split.py",
            "--dataset-name", "pusht",
            "--mock-data",
            "--max-samples", "64",
            "--seed", "42",
            "--output", str(self.output_path),
        ], capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(self.output_path.read_text())
        train_set = set(data["train_episode_ids"])
        eval_set = set(data["eval_episode_ids"])
        assert len(train_set & eval_set) == 0, "Train/eval overlap detected"
