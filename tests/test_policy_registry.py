"""Tests for policy registry and base interface."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from mini_vla.policies import (
    DeltaActionBCPolicy,
    HistoryBCPolicy,
    SingleFrameBCPolicy,
    build_policy,
    list_policies,
    register_policy,
)

# ── Minimal config that builds a valid MiniVLA model ────────────────────

BASE_MODEL_CFG: Dict[str, Any] = {
    "model": {
        "state_dim": 6,
        "action_dim": 2,
        "vision_encoder": {"type": "small_cnn", "output_dim": 128},
        "text_encoder": {"type": "mock_llm", "output_dim": 128},
        "state_encoder": {"output_dim": 128},
        "fusion": {"type": "concat_mlp", "input_dim": 384, "output_dim": 128},
        "action_head": {"input_dim": 128},
    },
    "data": {"batch_size": 2},
    "train": {"epochs": 1, "device": "cpu"},
}


def _dummy_batch(state_dim: int = 6):
    return {
        "image": torch.randn(2, 3, 64, 64),
        "input_ids": torch.randint(0, 128, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "state": torch.randn(2, state_dim),
        "action": torch.randn(2, 2),
    }


class TestPolicyRegistry:
    def test_known_policies_are_registered(self):
        names = list_policies()
        assert "single_frame_bc" in names
        assert "history_bc" in names
        assert "delta_action_bc" in names
        assert "action_chunk_bc" in names

    def test_build_policy_with_policy_section(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        assert isinstance(policy, SingleFrameBCPolicy)
        assert policy.policy_type == "single_frame_bc"

    def test_build_history_policy(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "history_bc"}}
        policy = build_policy(cfg)
        assert isinstance(policy, HistoryBCPolicy)

    def test_build_delta_policy(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "delta_action_bc"}}
        policy = build_policy(cfg)
        assert isinstance(policy, DeltaActionBCPolicy)

    def test_unknown_policy_raises(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "nonexistent"}}
        with pytest.raises(ValueError, match="Unknown policy type"):
            build_policy(cfg)

    def test_no_policy_type_raises(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy()}
        with pytest.raises(ValueError, match="No policy type"):
            build_policy(cfg)

    def test_register_unknown_class_raises(self):
        with pytest.raises(TypeError, match="must inherit from BasePolicy"):
            register_policy("bad", dict)  # type: ignore[arg-type]


class TestBasePolicyInterface:
    def test_compute_loss_returns_dict(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        batch = _dummy_batch()
        result = policy.compute_loss(batch)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "mse" in result
        assert "mae" in result
        assert result["loss"].dim() == 0  # scalar

    def test_predict_action_returns_dict(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        batch = _dummy_batch()
        result = policy.predict_action(batch)
        assert isinstance(result, dict)
        assert "action" in result
        assert result["action"].shape == (2, 2)

    def test_get_action_dim(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        assert policy.get_action_dim() == 2

    def test_supports_action_chunk_false(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        assert policy.supports_action_chunk() is False

    def test_device_property(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        assert str(policy.device) == "cpu"


class TestSingleFramePolicy:
    def test_forward_shape(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        batch = _dummy_batch()
        out = policy.compute_loss(batch)
        assert out["loss"].item() >= 0

    def test_predict_action_shape(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "single_frame_bc"}}
        policy = build_policy(cfg)
        batch = _dummy_batch()
        out = policy.predict_action(batch)
        assert out["action"].shape == (2, 2)


class TestHistoryPolicy:
    def test_forward_shape(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "history_bc"}}
        policy = build_policy(cfg)
        batch = _dummy_batch(state_dim=6)
        out = policy.compute_loss(batch)
        assert out["loss"].item() >= 0


class TestDeltaActionPolicy:
    def test_forward_shape(self):
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "delta_action_bc"}}
        policy = build_policy(cfg)
        batch = _dummy_batch(state_dim=6)
        out = policy.compute_loss(batch)
        assert out["loss"].item() >= 0

    def test_predict_action_reconstructs_action(self):
        """Delta policy should reconstruct pred_action = prev_action + pred_delta."""
        cfg = {"model": BASE_MODEL_CFG["model"].copy(), "policy": {"type": "delta_action_bc"}}
        policy = build_policy(cfg)
        batch = _dummy_batch(state_dim=6)
        # Add prev_action for reconstruction
        batch["prev_action"] = torch.full((2, 2), 10.0)
        batch["prev_action_normalized"] = torch.full((2, 2), 10.0)
        out = policy.predict_action(batch)
        assert out["action"].shape == (2, 2)
        # Action should be close to prev_action (model predicts near-zero delta)
        assert torch.allclose(out["action"].mean(), torch.tensor(10.0), atol=5.0)
