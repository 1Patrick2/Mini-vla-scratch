"""Tests for split-aware stats computation."""


import numpy as np

from mini_vla.datasets.normalization import compute_stats
from mini_vla.datasets.splits import create_episode_split, filter_samples_by_episode


def _make_samples_with_extreme_eval():
    """Train episodes have small actions, eval episodes have extreme actions."""
    rng = np.random.RandomState(42)
    samples = []
    for i in range(32):
        ep = i // 4
        # Make episodes 6-7 (eval) have extreme action values
        if ep >= 6:
            action = [100.0, 100.0]
        else:
            action = [1.0, 1.0]
        samples.append({
            "state": rng.randn(2).astype(np.float32),
            "action": np.array(action, dtype=np.float32),
            "episode_index": ep,
            "frame_index": i % 4,
        })
    return samples


class TestSplitAwareStats:
    def test_train_only_stats_not_polluted_by_eval(self):
        samples = _make_samples_with_extreme_eval()
        split = create_episode_split(samples, dataset_name="test", train_ratio=0.8, seed=42)
        train_samples = filter_samples_by_episode(samples, split.train_episode_ids)
        eval_samples = filter_samples_by_episode(samples, split.eval_episode_ids)

        train_stats = compute_stats(train_samples, state_keys=["state"], action_keys=["action"])
        eval_stats = compute_stats(eval_samples, state_keys=["state"], action_keys=["action"])

        # Train and eval have different action scales
        # Verify they are genuinely different (not the same pool)
        assert abs(train_stats.action_mean[0] - eval_stats.action_mean[0]) > 10.0, (
            f"Train ({train_stats.action_mean[0]}) and eval ({eval_stats.action_mean[0]}) "
            f"action means should differ"
        )
