"""Tests for Toy 2D data generation."""

import json
from pathlib import Path

import pytest

from scripts.generate_toy_data import generate_toy_data


class TestGenerateToyData:
    """End-to-end data generation verification."""

    def test_generates_correct_structure(self, tmp_path):
        root = generate_toy_data(
            output_root=tmp_path,
            num_episodes=2,
            max_steps=5,
            image_size=64,
            seed=42,
        )

        # Episode dirs exist
        ep0 = root / "episodes" / "ep_000000"
        ep1 = root / "episodes" / "ep_000001"
        assert ep0.is_dir()
        assert ep1.is_dir()

        # episode.json exists and has required fields
        for ep_dir in [ep0, ep1]:
            js = ep_dir / "episode.json"
            assert js.exists()
            with open(js, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "episode_id" in data
            assert "instruction" in data
            assert "steps" in data
            assert len(data["steps"]) == 5

            for step in data["steps"]:
                assert "frame" in step
                assert "state" in step
                assert "action" in step
                assert "target" in step
                assert len(step["state"]) == 2
                assert len(step["action"]) == 2
                assert len(step["target"]) == 2

        # Frame PNG files exist
        for ep_dir in [ep0, ep1]:
            for step_idx in range(5):
                png = ep_dir / "frames" / f"{step_idx:06d}.png"
                assert png.exists(), f"Missing frame: {png}"
                assert png.stat().st_size > 100  # not empty

    def test_episodes_are_deterministic(self, tmp_path):
        root1 = generate_toy_data(tmp_path / "a", num_episodes=1, max_steps=3, image_size=64, seed=42)
        root2 = generate_toy_data(tmp_path / "b", num_episodes=1, max_steps=3, image_size=64, seed=42)

        with open(root1 / "episodes" / "ep_000000" / "episode.json") as f:
            ep1 = json.load(f)
        with open(root2 / "episodes" / "ep_000000" / "episode.json") as f:
            ep2 = json.load(f)

        assert ep1["steps"] == ep2["steps"]
