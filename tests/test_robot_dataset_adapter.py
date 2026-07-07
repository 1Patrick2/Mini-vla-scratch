"""Tests for the generic robot dataset adapter."""

import numpy as np
import pytest
import torch

from mini_vla.datasets.normalization import ActionNormalizer, NormalizationStats
from mini_vla.datasets.robot_adapter import BaseRobotDatasetAdapter
from mini_vla.datasets.spec import DatasetSpec

_PUSHT_SPEC = DatasetSpec(
    name="pusht",
    repo_id="lerobot/pusht",
    image_keys=["observation.image", "image"],
    state_keys=["observation.state", "state"],
    action_keys=["action"],
    language_keys=["task", "language_instruction", "instruction"],
    default_instruction="push the T block",
)


def _flat_sample():
    """Return a flat-key sample like current PushT."""
    return {
        "observation.image": np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8),
        "observation.state": np.random.randn(2).astype(np.float32),
        "action": np.random.randn(2).astype(np.float32),
        "episode_index": 0,
        "frame_index": 0,
    }


def _nested_sample():
    """Return a nested-key sample."""
    return {
        "observation": {
            "image": np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "state": np.random.randn(2).astype(np.float32),
        },
        "action": np.random.randn(2).astype(np.float32),
        "episode_index": 0,
        "frame_index": 0,
    }


class TestBaseRobotDatasetAdapter:
    def test_flat_keys(self):
        adapter = BaseRobotDatasetAdapter([_flat_sample()], _PUSHT_SPEC)
        sample = adapter[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["state"].shape == (2,)
        assert sample["action"].shape == (2,)
        assert "instruction" in sample

    def test_nested_keys(self):
        adapter = BaseRobotDatasetAdapter([_nested_sample()], _PUSHT_SPEC)
        sample = adapter[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["state"].shape == (2,)
        assert sample["action"].shape == (2,)

    def test_default_instruction_fallback(self):
        sample = {
            "observation.image": np.zeros((96, 96, 3), dtype=np.uint8),
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
        }
        adapter = BaseRobotDatasetAdapter([sample], _PUSHT_SPEC)
        assert adapter[0]["instruction"] == "push the T block"

    def test_missing_image_strict_raises(self):
        sample = {"action": np.zeros(2, dtype=np.float32)}
        with pytest.raises(KeyError, match="image key"):
            BaseRobotDatasetAdapter([sample], _PUSHT_SPEC, strict=True)[0]

    def test_missing_action_strict_raises(self):
        sample = {
            "observation.image": np.zeros((96, 96, 3), dtype=np.uint8),
            "observation.state": np.zeros(2, dtype=np.float32),
        }
        with pytest.raises(KeyError, match="action key"):
            BaseRobotDatasetAdapter([sample], _PUSHT_SPEC, strict=True)[0]

    def test_high_dim_action(self):
        """Adapter should handle action_dim > 2."""
        spec = DatasetSpec(
            name="test",
            repo_id="lerobot/test",
            image_keys=["image"],
            state_keys=["state"],
            action_keys=["action"],
        )
        sample = {
            "image": np.zeros((96, 96, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
            "action": np.zeros(14, dtype=np.float32),
        }
        adapter = BaseRobotDatasetAdapter([sample], spec)
        out = adapter[0]
        assert out["action"].shape == (14,)
        assert out["state"].shape == (8,)

    def test_normalizer_applied(self):
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.ones(2) * 5, action_std=torch.ones(2) * 2,
        )
        normalizer = ActionNormalizer(stats)
        # Use a deterministic action value
        sample = {
            "observation.image": np.zeros((96, 96, 3), dtype=np.uint8),
            "observation.state": np.array([1.0, 2.0], dtype=np.float32),
            "action": np.array([7.0, 9.0], dtype=np.float32),
        }
        adapter = BaseRobotDatasetAdapter(
            [sample], _PUSHT_SPEC, normalizer=normalizer,
        )
        out = adapter[0]
        # Action was normalized: (7-5)/2 = 1.0, (9-5)/2 = 2.0
        expected_action = torch.tensor([1.0, 2.0])
        assert torch.allclose(out["action"], expected_action, atol=1e-5)
