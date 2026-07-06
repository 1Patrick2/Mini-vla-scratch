"""Tests for PushT dataset inspection utilities."""

import numpy as np
import pytest

from mini_vla.datasets.pusht_inspection import (
    build_pusht_report,
    compute_state_action_stats,
    discover_feature_keys,
    summarize_pusht_schema,
)


def _make_mock_sample(seed=0):
    """Return a mock PushT-like sample dict."""
    rng = np.random.RandomState(seed)
    return {
        "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
        "observation.state": rng.randn(2).astype(np.float32),
        "action": rng.randn(2).astype(np.float32),
        "episode_index": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "next.reward": 0.0,
        "next.done": False,
        "next.success": False,
        "task_index": 0,
    }


class TestDiscoverFeatureKeys:
    def test_returns_defaults_without_sample(self):
        keys = discover_feature_keys()
        assert keys["image_key"] == "observation.image"
        assert keys["state_key"] == "observation.state"
        assert keys["action_key"] == "action"

    def test_raises_on_missing_key(self):
        with pytest.raises(KeyError):
            discover_feature_keys(sample={"bad": 1})

    def test_passes_with_valid_sample(self):
        sample = _make_mock_sample()
        keys = discover_feature_keys(sample)
        assert keys["image_key"] in sample

    def test_required_only_passes(self):
        """Sample with only required keys (no metadata) should pass."""
        sample = {
            "observation.image": np.zeros((96, 96, 3), dtype=np.uint8),
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
        }
        keys = discover_feature_keys(sample)
        assert keys["image_key"] == "observation.image"


class TestSummarizePushtSchema:
    def test_empty_samples(self):
        schema = summarize_pusht_schema([])
        assert schema["num_samples"] == 0

    def test_basic_fields(self):
        sample = _make_mock_sample()
        schema = summarize_pusht_schema([sample])
        assert schema["num_samples"] == 1
        assert schema["image_shape"] == [96, 96, 3]
        assert schema["state_shape"] == [2]
        assert schema["action_shape"] == [2]
        assert schema["has_reward"] is True
        assert schema["has_episode_index"] is True
        assert schema["has_timestamp"] is True

    def test_nested_sample_schema(self):
        """Nested observation dict should still produce correct schema."""
        rng = np.random.RandomState(1)
        sample = {
            "observation": {
                "image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
                "state": rng.randn(2).astype(np.float32),
            },
            "action": rng.randn(2).astype(np.float32),
        }
        schema = summarize_pusht_schema([sample])
        assert schema["image_shape"] == [96, 96, 3]
        assert schema["state_shape"] == [2]
        assert schema["action_shape"] == [2]

    def test_optional_metadata_missing(self):
        """When optional metadata is absent, has_* fields should be False."""
        sample = {
            "observation.image": np.zeros((96, 96, 3), dtype=np.uint8),
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
        }
        schema = summarize_pusht_schema([sample])
        assert schema["has_reward"] is False
        assert schema["has_done"] is False
        assert schema["has_success"] is False
        assert schema["has_episode_index"] is False


class TestComputeStateActionStats:
    def test_empty_samples(self):
        stats = compute_state_action_stats([])
        assert stats["state"] == {}
        assert stats["action"] == {}

    def test_stats_shapes(self):
        samples = [_make_mock_sample(i) for i in range(10)]
        stats = compute_state_action_stats(samples, max_samples=5)
        assert "min" in stats["state"]
        assert "max" in stats["state"]
        assert "mean" in stats["state"]
        assert "std" in stats["state"]
        assert len(stats["state"]["min"]) == 2

    def test_action_stats(self):
        samples = [_make_mock_sample(i) for i in range(10)]
        stats = compute_state_action_stats(samples)
        assert "min" in stats["action"]
        assert len(stats["action"]["mean"]) == 2

    def test_single_sample_std_not_nan(self):
        """Single-sample std should be [0.0, 0.0], not NaN."""
        sample = _make_mock_sample()
        stats = compute_state_action_stats([sample])
        for std_val in stats["state"]["std"]:
            assert std_val == 0.0, f"State std {std_val} should be 0.0 (not NaN)"
        for std_val in stats["action"]["std"]:
            assert std_val == 0.0, f"Action std {std_val} should be 0.0 (not NaN)"


class TestBuildPushtReport:
    def test_report_contains_all_keys(self):
        samples = [_make_mock_sample(i) for i in range(5)]
        report = build_pusht_report(samples, repo_id="lerobot/pusht", max_samples=3)
        assert report["repo_id"] == "lerobot/pusht"
        assert report["num_samples"] == 5
        assert "image_shape" in report
        assert "state_shape" in report
        assert "action_shape" in report
        assert "state_stats" in report
        assert "action_stats" in report
