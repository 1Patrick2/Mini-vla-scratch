"""Tests for the factory with robot_dataset type."""

import numpy as np
import pytest

from mini_vla.datasets.factory import build_dataset
from mini_vla.datasets.robot_adapter import BaseRobotDatasetAdapter


def _make_mock_push_t_samples(n=4):
    rng = np.random.RandomState(42)
    return [
        {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": rng.randn(2).astype(np.float32),
            "action": rng.randn(2).astype(np.float32),
            "episode_index": i // 2,
            "frame_index": i % 2,
        }
        for i in range(n)
    ]


class TestBuildRobotDataset:
    def test_returns_adapter(self):
        samples = _make_mock_push_t_samples()
        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
            "image_size": 64,
            "max_text_len": 16,
        }
        ds = build_dataset(cfg, base_dataset=samples)
        assert isinstance(ds, BaseRobotDatasetAdapter)
        assert len(ds) == 4

    def test_sample_keys(self):
        samples = _make_mock_push_t_samples()
        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
        }
        ds = build_dataset(cfg, base_dataset=samples)
        sample = ds[0]
        expected = {"image", "input_ids", "attention_mask", "state", "action", "instruction"}
        assert expected.issubset(sample.keys())

    def test_image_shape(self):
        samples = _make_mock_push_t_samples()
        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
            "image_size": 64,
        }
        ds = build_dataset(cfg, base_dataset=samples)
        assert ds[0]["image"].shape == (3, 64, 64)

    def test_unknown_dataset_raises(self):
        samples = _make_mock_push_t_samples()
        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "nonexistent",
        }
        with pytest.raises(ValueError, match="Unknown"):
            build_dataset(cfg, base_dataset=samples)

    def test_missing_stats_path_raises(self):
        samples = _make_mock_push_t_samples()
        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
            "normalization": {
                "enabled": True,
                "stats_path": "nonexistent_stats.json",
            },
        }
        with pytest.raises(FileNotFoundError, match="stats_path"):
            build_dataset(cfg, base_dataset=samples)
