"""Tests for Predictor with delta action policy — verifies prev_action passthrough."""

from __future__ import annotations

from typing import Any, Dict

import torch

from mini_vla.inference.predictor import Predictor
from mini_vla.models import build_model
from mini_vla.training.checkpoint import save_checkpoint

# ── Minimal config for a delta_action_bc policy ────────────────────────

DELTA_CFG: Dict[str, Any] = {
    "model": {
        "state_dim": 6,
        "action_dim": 2,
        "vision_encoder": {"type": "small_cnn", "output_dim": 16},
        "text_encoder": {"type": "mock_llm", "output_dim": 16},
        "state_encoder": {"output_dim": 16},
        "fusion": {"type": "concat_mlp", "input_dim": 48, "output_dim": 16},
        "action_head": {"input_dim": 16},
    },
    "policy": {"type": "delta_action_bc"},
    "train": {"device": "cpu"},
}


class TestPredictorDeltaPassthrough:
    def test_predictor_builds_with_delta_policy(self, tmp_path):
        """Predictor should build a DeltaActionBCPolicy when config specifies it."""
        ckpt = tmp_path / "test.pt"
        # Create a minimal model checkpoint
        from mini_vla.policies import build_policy

        policy = build_policy(DELTA_CFG)
        save_checkpoint(ckpt, model=policy.model, epoch=1, metrics={"loss": 0.0}, config=DELTA_CFG)

        predictor = Predictor(DELTA_CFG, ckpt, device="cpu", clip_action=False)
        assert predictor._policy is not None
        assert predictor._policy.policy_type == "delta_action_bc"

    def test_predict_passes_prev_action_to_policy(self, tmp_path):
        """Predictor should pass prev_action from sample to the policy batch."""
        ckpt = tmp_path / "test2.pt"
        # Save a minimal checkpoint
        from mini_vla.models import build_model
        model = build_model(DELTA_CFG)
        save_checkpoint(ckpt, model=model, epoch=1, metrics={"loss": 0.0}, config=DELTA_CFG)

        predictor = Predictor(DELTA_CFG, ckpt, device="cpu", clip_action=False)

        # Create a sample with prev_action
        sample = {
            "image": torch.randn(3, 64, 64),
            "input_ids": torch.randint(0, 128, (16,)),
            "attention_mask": torch.ones(16, dtype=torch.long),
            "state": torch.randn(6),
            "prev_action": torch.tensor([10.0, 10.0]),
            "prev_action_normalized": torch.tensor([10.0, 10.0]),
        }

        action = predictor.predict(sample)
        assert action.shape == (2,)
        # The predicted action should be close to prev_action (model predicts near-zero delta)
        # Since model is untrained, this is a sanity check
        assert torch.isfinite(action).all()

    def test_zero_delta_reconstructs_prev_action(self, tmp_path):
        """When model predicts zero delta, predict() should return prev_action."""
        ckpt = tmp_path / "test3.pt"
        model = build_model(DELTA_CFG)
        save_checkpoint(ckpt, model=model, epoch=1, metrics={"loss": 0.0}, config=DELTA_CFG)

        predictor = Predictor(DELTA_CFG, ckpt, device="cpu", clip_action=False)

        # Monkey-patch the policy's model to return zero delta
        def zero_delta_forward(batch):
            batch_size = batch["state"].shape[0]
            return torch.zeros(batch_size, 2)

        predictor._policy.model.forward = zero_delta_forward

        sample = {
            "image": torch.randn(3, 64, 64),
            "input_ids": torch.randint(0, 128, (16,)),
            "attention_mask": torch.ones(16, dtype=torch.long),
            "state": torch.randn(6),
            "prev_action": torch.tensor([10.0, 10.0]),
        }

        action = predictor.predict(sample)
        assert torch.allclose(action, torch.tensor([10.0, 10.0]), atol=1e-6)
