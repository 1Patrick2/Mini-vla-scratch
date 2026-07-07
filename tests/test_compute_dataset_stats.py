"""Tests for the compute_dataset_stats CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestComputeDatasetStatsCLI:
    """Subprocess tests with --mock-data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.output_path = tmp_path / "stats.json"

    def _run(self, *args):
        cmd = [
            sys.executable,
            "scripts/compute_dataset_stats.py",
            *args,
            "--output", str(self.output_path),
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    def test_mock_pusht(self):
        result = self._run("--dataset-name", "pusht", "--mock-data", "--max-samples", "16")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists()
        data = json.loads(self.output_path.read_text())
        assert "state_mean" in data
        assert "action_mean" in data
        assert len(data["state_mean"]) == 2
        assert len(data["action_mean"]) == 2
        assert "state_std" in data
        assert "action_std" in data

    def test_unknown_dataset(self):
        result = self._run("--dataset-name", "nonexistent", "--mock-data")
        assert result.returncode != 0

    def test_stdout_contains_stats(self):
        result = self._run("--dataset-name", "pusht", "--mock-data", "--max-samples", "8")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "state_mean" in result.stdout
        assert "action_mean" in result.stdout
