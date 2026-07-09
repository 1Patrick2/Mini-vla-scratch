"""Tests for collate_toy_2d — verifies passthrough of additional fields."""

from __future__ import annotations

import torch

from mini_vla.datasets.collate import collate_toy_2d


def _make_chunk_sample(
    action_chunk_shape=(4, 2),
    has_prev=True,
    has_meta=True,
):
    """Create a mock sample mimicking ActionChunkTargetWrapper output."""
    sample = {
        "image": torch.randn(3, 64, 64),
        "input_ids": torch.randint(0, 128, (16,)),
        "attention_mask": torch.ones(16, dtype=torch.long),
        "state": torch.randn(6),
        "action": torch.randn(2),
        "action_chunk": torch.randn(*action_chunk_shape),
        "target_type": "action_chunk",
        "action_horizon": 4,
    }
    if has_prev:
        sample["prev_action"] = torch.randn(2)
        sample["prev_state"] = torch.randn(2)
    return sample


class TestCollateActionChunk:
    def test_preserves_action_chunk(self):
        batch = [self._make_chunk() for _ in range(4)]
        out = collate_toy_2d(batch)
        assert "action_chunk" in out
        assert out["action_chunk"].shape == (4, 4, 2)

    def _make_chunk(self):
        return _make_chunk_sample()

    def test_preserves_prev_fields(self):
        batch = [self._make_chunk() for _ in range(2)]
        out = collate_toy_2d(batch)
        assert "prev_action" in out
        assert "prev_state" in out
        assert out["prev_action"].shape == (2, 2)
        assert out["prev_state"].shape == (2, 2)

    def test_preserves_target_type_as_list(self):
        batch = [self._make_chunk() for _ in range(3)]
        out = collate_toy_2d(batch)
        assert "target_type" in out
        assert isinstance(out["target_type"], list)
        assert out["target_type"] == ["action_chunk", "action_chunk", "action_chunk"]

    def test_action_horizon_as_tensor(self):
        batch = [self._make_chunk() for _ in range(2)]
        out = collate_toy_2d(batch)
        assert "action_horizon" in out
        assert isinstance(out["action_horizon"], torch.Tensor)
        assert out["action_horizon"].tolist() == [4, 4]

    def test_standard_keys_still_present(self):
        batch = [self._make_chunk() for _ in range(2)]
        out = collate_toy_2d(batch)
        assert "image" in out
        assert "input_ids" in out
        assert "attention_mask" in out
        assert "state" in out
        assert "action" in out

    def test_variable_length_tensors_kept_as_list(self):
        """If one sample lacks a tensor field, others should not crash."""
        batch = [
            _make_chunk_sample(has_prev=True),
            _make_chunk_sample(has_prev=False),
        ]
        out = collate_toy_2d(batch)
        # prev_action should appear for the first sample
        assert "prev_action" in out
        # The first sample has prev_action, second doesn't — list is expected
        assert isinstance(out["prev_action"], list)
        assert out["prev_action"][0] is not None
        assert out["prev_action"][1] is None

    def test_raw_normalized_chunks(self):
        sample = _make_chunk_sample()
        sample["action_chunk_raw"] = torch.randn(4, 2)
        sample["action_chunk_normalized"] = torch.randn(4, 2)
        batch = [sample for _ in range(3)]
        out = collate_toy_2d(batch)
        assert out["action_chunk_raw"].shape == (3, 4, 2)
        assert out["action_chunk_normalized"].shape == (3, 4, 2)
