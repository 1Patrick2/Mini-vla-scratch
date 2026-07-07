"""Tests for the generic robot dataset inspect CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mini_vla.datasets.robot_inspection import (
    inspect_robot_dataset_from_registry,
)


class TestInspectRobotDatasetCLI:
    """Subprocess tests with --mock-data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.output_path = tmp_path / "report.json"

    def _run(self, *args):
        cmd = [
            sys.executable,
            "scripts/inspect_robot_dataset.py",
            *args,
            "--output", str(self.output_path),
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    def test_mock_pusht(self):
        result = self._run("--dataset-name", "pusht", "--mock-data")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists()
        report = json.loads(self.output_path.read_text())
        assert report["dataset_name"] == "pusht"
        assert report["matched_image_key"] is not None
        assert "vision OK" in report["matched_summary"]

    def test_mock_aloha(self):
        result = self._run("--dataset-name", "aloha_sim_transfer_cube", "--mock-data")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists()
        report = json.loads(self.output_path.read_text())
        assert report["dataset_name"] == "aloha_sim_transfer_cube"

    def test_mock_libero(self):
        result = self._run("--dataset-name", "libero", "--mock-data")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert self.output_path.exists()
        report = json.loads(self.output_path.read_text())
        assert report["dataset_name"] == "libero"

    def test_unknown_dataset(self):
        result = self._run("--dataset-name", "nonexistent", "--mock-data")
        assert result.returncode != 0

    def test_mock_report_content(self):
        """Inspect report should contain expected keys."""
        report = inspect_robot_dataset_from_registry("pusht")
        assert "matched_image_key" in report
        assert "matched_state_key" in report
        assert "matched_action_key" in report
        assert "image_shape" in report
        assert "available_keys" in report
        assert "repo_id" in report
