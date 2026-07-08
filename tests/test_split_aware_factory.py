"""Tests for split-aware dataset factory."""

import numpy as np

from mini_vla.datasets.factory import build_dataset
from mini_vla.datasets.robot_adapter import BaseRobotDatasetAdapter
from mini_vla.datasets.splits import create_episode_split, save_split


def _make_samples(n=32):
    rng = np.random.RandomState(42)
    return [
        {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": rng.randn(2).astype(np.float32),
            "action": rng.randn(2).astype(np.float32),
            "episode_index": i // 4,
            "frame_index": i % 4,
        }
        for i in range(n)
    ]


class TestSplitAwareFactory:
    def test_split_train_only_returns_train_samples(self, tmp_path):
        samples = _make_samples()
        split = create_episode_split(samples, dataset_name="pusht", train_ratio=0.8)
        split_path = save_split(split, tmp_path / "split.json")

        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
            "split": {
                "enabled": True,
                "path": str(split_path),
                "name": "train",
            },
        }
        ds = build_dataset(cfg, base_dataset=samples)
        assert isinstance(ds, BaseRobotDatasetAdapter)
        # All samples should belong to train episodes
        for i in range(len(ds)):
            assert ds[i]["episode_index"] in split.train_episode_ids

    def test_split_eval_only_returns_eval_samples(self, tmp_path):
        samples = _make_samples()
        split = create_episode_split(samples, dataset_name="pusht", train_ratio=0.8)
        split_path = save_split(split, tmp_path / "split.json")

        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
            "split": {
                "enabled": True,
                "path": str(split_path),
                "name": "eval",
            },
        }
        ds = build_dataset(cfg, base_dataset=samples)
        assert isinstance(ds, BaseRobotDatasetAdapter)
        for i in range(len(ds)):
            assert ds[i]["episode_index"] in split.eval_episode_ids

    def test_split_disabled_returns_all(self, tmp_path):
        samples = _make_samples()
        cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
        }
        ds = build_dataset(cfg, base_dataset=samples)
        assert len(ds) == len(samples)

    def test_split_no_overlap(self, tmp_path):
        samples = _make_samples()
        split = create_episode_split(samples, dataset_name="pusht", train_ratio=0.8)
        split_path = save_split(split, tmp_path / "split.json")

        train_cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
            "split": {"enabled": True, "path": str(split_path), "name": "train"},
        }
        eval_cfg = {
            "dataset_type": "robot_dataset",
            "dataset_name": "pusht",
            "split": {"enabled": True, "path": str(split_path), "name": "eval"},
        }
        train_ds = build_dataset(train_cfg, base_dataset=samples)
        eval_ds = build_dataset(eval_cfg, base_dataset=samples)

        train_eps = {train_ds[i]["episode_index"] for i in range(len(train_ds))}
        eval_eps = {eval_ds[i]["episode_index"] for i in range(len(eval_ds))}
        assert len(train_eps & eval_eps) == 0
