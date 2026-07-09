"""Tests for high-dimensional policy support (state_dim=14, action_dim=14).

These tests validate that policies work with dimensions similar to ALOHA
(longer than PushT's 2), catching hardcoded-dimension assumptions.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from mini_vla.policies import build_policy

# ── High-dim config mimicking ALOHA (14-dim state and action) ──────────

HIGH_DIM_CFG: Dict[str, Any] = {
    "model": {
        "state_dim": 14,
        "action_dim": 14,
        "vision_encoder": {"type": "small_cnn", "output_dim": 16},
        "text_encoder": {"type": "mock_llm", "output_dim": 16},
        "state_encoder": {"output_dim": 16},
        "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
        "action_head": {"input_dim": 16},
    },
    "policy": {"type": None},  # filled per test
    "train": {"device": "cpu"},
}


def _high_dim_batch(state_dim: int = 14, action_dim: int = 14):
    return {
        "image": torch.randn(2, 3, 64, 64),
        "input_ids": torch.randint(0, 128, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "state": torch.randn(2, state_dim),
        "action": torch.randn(2, action_dim),
    }


class TestHighDimSingleFrame:
    def test_compute_loss(self):
        cfg = HIGH_DIM_CFG.copy()
        cfg["policy"] = {"type": "single_frame_bc"}
        policy = build_policy(cfg)
        batch = _high_dim_batch()
        out = policy.compute_loss(batch)
        assert out["loss"].item() >= 0

    def test_predict_action_shape(self):
        cfg = HIGH_DIM_CFG.copy()
        cfg["policy"] = {"type": "single_frame_bc"}
        policy = build_policy(cfg)
        batch = _high_dim_batch()
        out = policy.predict_action(batch)
        assert out["action"].shape == (2, 14)

    def test_action_dim_via_shape_meta(self):
        """If shape_meta.action.dim is set, policy should use it."""
        cfg = HIGH_DIM_CFG.copy()
        cfg["shape_meta"] = {"action": {"dim": 14}, "state": {"dim": 14}}
        cfg["policy"] = {"type": "single_frame_bc"}
        policy = build_policy(cfg)
        assert policy.get_action_dim() == 14


class TestHighDimHistory:
    def test_compute_loss(self):
        cfg = HIGH_DIM_CFG.copy()
        cfg["model"]["state_dim"] = 14 + 14 + 14  # state + prev_state + prev_action
        cfg["policy"] = {"type": "history_bc"}
        policy = build_policy(cfg)
        batch = _high_dim_batch(state_dim=42)
        out = policy.compute_loss(batch)
        assert out["loss"].item() >= 0


class TestHighDimDelta:
    def test_compute_loss(self):
        cfg = HIGH_DIM_CFG.copy()
        cfg["model"]["state_dim"] = 14 + 14 + 14  # state + prev_state + prev_action
        cfg["policy"] = {"type": "delta_action_bc"}
        policy = build_policy(cfg)
        batch = _high_dim_batch(state_dim=42, action_dim=14)
        out = policy.compute_loss(batch)
        assert out["loss"].item() >= 0

    def test_predict_action_reconstructs_high_dim(self):
        cfg = HIGH_DIM_CFG.copy()
        cfg["model"]["state_dim"] = 42
        cfg["policy"] = {"type": "delta_action_bc"}
        policy = build_policy(cfg)
        batch = _high_dim_batch(state_dim=42)
        batch["prev_action"] = torch.full((2, 14), 5.0)
        batch["prev_action_normalized"] = torch.full((2, 14), 5.0)
        out = policy.predict_action(batch)
        assert out["action"].shape == (2, 14)
        # Action should be near prev_action since model predicts near-zero delta
        assert out["action"].mean().item() == pytest.approx(5.0, abs=5.0)
