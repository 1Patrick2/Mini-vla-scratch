"""Tests for ActionChunkTargetWrapper — chunking and episode boundary."""

from __future__ import annotations

import torch

from mini_vla.datasets.transforms import ActionChunkTargetWrapper


def _make_episode_dataset(
    num_episodes: int = 2,
    frames_per_ep: int = 6,
    action_dim: int = 2,
):
    """Create a mock dataset with episode_index and action."""
    samples = []
    idx = 0
    for ep in range(num_episodes):
        for fr in range(frames_per_ep):
            samples.append({
                "episode_index": ep,
                "frame_index": fr,
                "action": torch.tensor([float(ep * 100 + fr), float(ep * 100 + fr + 1)]),
                "state": torch.randn(2),
            })
            idx += 1
    return samples


class TestActionChunkTargetWrapper:
    def test_basic_chunk_shape(self):
        ds = _make_episode_dataset(frames_per_ep=6)
        wrapper = ActionChunkTargetWrapper(ds, action_horizon=4)
        sample = wrapper[0]
        assert sample["action_chunk"].shape == (4, 2)
        assert sample["target_type"] == "action_chunk"
        assert sample["action_horizon"] == 4

    def test_chunk_target_values(self):
        """Each chunk step should match the corresponding future action."""
        ds = _make_episode_dataset(frames_per_ep=5)
        wrapper = ActionChunkTargetWrapper(ds, action_horizon=3)
        sample = wrapper[0]  # frame 0 -> chunk [0, 1, 2]
        for step in range(3):
            expected = ds[step]["action"]
            assert torch.allclose(
                sample["action_chunk"][step], expected
            ), f"Step {step}: expected {expected}, got {sample['action_chunk'][step]}"

    def test_episode_boundary_respected(self):
        """Chunk must not cross episode boundary."""
        ds = _make_episode_dataset(num_episodes=2, frames_per_ep=4)
        wrapper = ActionChunkTargetWrapper(ds, action_horizon=4)
        # Each episode has exactly 4 frames, so H=4 leaves 1 valid start per ep
        assert len(wrapper) == 2
        # First chunk is entirely within ep 0
        s0 = wrapper[0]
        assert s0["episode_index"] == 0
        # Second chunk is entirely within ep 1
        s1 = wrapper[1]
        assert s1["episode_index"] == 1

    def test_tail_frames_dropped(self):
        """Samples within H-1 of episode end should be excluded."""
        ds = _make_episode_dataset(num_episodes=1, frames_per_ep=5)
        wrapper = ActionChunkTargetWrapper(ds, action_horizon=3)
        # ep has 5 frames -> valid starts: [0, 1, 2] (3 valid)
        assert len(wrapper) == 3
        # original indices 3,4 should be dropped
        valid_indices = set(wrapper._valid_indices)
        assert 3 not in valid_indices
        assert 4 not in valid_indices

    def test_len_with_horizon_1(self):
        ds = _make_episode_dataset(frames_per_ep=5)
        wrapper = ActionChunkTargetWrapper(ds, action_horizon=1)
        assert len(wrapper) == 10  # all frames valid

    def test_invalid_horizon_raises(self):
        ds = _make_episode_dataset()
        import pytest
        with pytest.raises(ValueError, match="action_horizon"):
            ActionChunkTargetWrapper(ds, action_horizon=0)
