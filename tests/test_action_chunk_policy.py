"""Tests for ActionChunkBCPolicy — forward, loss, and predict."""

from __future__ import annotations

from typing import Any, Dict

import torch

from mini_vla.policies import build_policy

# ── Config mimicking PushT with action_horizon=4 ────────────────────────

CHUNK_CFG: Dict[str, Any] = {
    "model": {
        "state_dim": 2,
        "action_dim": 8,  # flattened: 4 * 2
        "vision_encoder": {"type": "small_cnn", "output_dim": 16},
        "text_encoder": {"type": "mock_llm", "output_dim": 16},
        "state_encoder": {"output_dim": 16},
        "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
        "action_head": {"input_dim": 16},
    },
    "policy": {"type": "action_chunk_bc"},
    "shape_meta": {
        "action": {"dim": 2},
        "state": {"dim": 2},
        "action_horizon": 4,
    },
    "train": {"device": "cpu"},
}


def _chunk_batch(batch_size: int = 2):
    return {
        "image": torch.randn(batch_size, 3, 64, 64),
        "input_ids": torch.randint(0, 128, (batch_size, 16)),
        "attention_mask": torch.ones(batch_size, 16, dtype=torch.long),
        "state": torch.randn(batch_size, 2),
        "action_chunk": torch.randn(batch_size, 4, 2),
    }


class TestActionChunkPolicy:
    def test_builds(self):
        policy = build_policy(CHUNK_CFG)
        assert policy.policy_type == "action_chunk_bc"
        assert policy.action_horizon == 4
        assert policy.step_action_dim == 2
        assert policy.supports_action_chunk()

    def test_compute_loss_shape(self):
        policy = build_policy(CHUNK_CFG)
        batch = _chunk_batch()
        out = policy.compute_loss(batch)
        assert "loss" in out
        assert "mse" in out
        assert "mae" in out
        assert "mae_first" in out
        assert out["loss"].dim() == 0

    def test_predict_action_shape(self):
        policy = build_policy(CHUNK_CFG)
        batch = _chunk_batch()
        out = policy.predict_action(batch)
        assert "action_chunk" in out
        assert "action" in out
        assert out["action_chunk"].shape == (2, 4, 2)
        assert out["action"].shape == (2, 2)
        # action should be first step of chunk
        assert torch.allclose(out["action"], out["action_chunk"][:, 0])

    def test_wrong_flattened_dim_raises(self):
        """If model.action_dim doesn't match H * D, reshape should raise."""
        bad_cfg: Dict[str, Any] = {
            "model": {
                "state_dim": 2,
                "action_dim": 6,  # wrong: should be 4*2=8
                "vision_encoder": {"type": "small_cnn", "output_dim": 16},
                "text_encoder": {"type": "mock_llm", "output_dim": 16},
                "state_encoder": {"output_dim": 16},
                "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
                "action_head": {"input_dim": 16},
            },
            "policy": {"type": "action_chunk_bc"},
            "shape_meta": {
                "action": {"dim": 2},
                "state": {"dim": 2},
                "action_horizon": 4,
            },
            "train": {"device": "cpu"},
        }
        import pytest
        policy = build_policy(bad_cfg)  # should succeed with wrong dim
        batch = _chunk_batch()
        with pytest.raises(ValueError, match="Expected flattened dim"):
            policy.compute_loss(batch)

    def test_reshaped_output_equals_flat(self):
        """Verify that _reshape_chunk correctly recovers the original values."""
        policy = build_policy(CHUNK_CFG)
        flat = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        chunk = policy._reshape_chunk(flat)
        assert chunk.shape == (1, 4, 2)
        assert chunk[0, 0].tolist() == [1.0, 2.0]
        assert chunk[0, 1].tolist() == [3.0, 4.0]
        assert chunk[0, 2].tolist() == [5.0, 6.0]
        assert chunk[0, 3].tolist() == [7.0, 8.0]
