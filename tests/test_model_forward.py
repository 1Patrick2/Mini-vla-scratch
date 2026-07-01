"""Tests for MiniVLA end-to-end forward."""

import torch

from mini_vla.models import (
    ActionHead,
    FusionMLP,
    MiniVLA,
    MockLLMTextEncoder,
    SmallCNNVisionEncoder,
    StateEncoder,
)


def _build_debug_mini_vla() -> MiniVLA:
    """Build a MiniVLA with default component shapes for testing."""
    return MiniVLA(
        vision_encoder=SmallCNNVisionEncoder(output_dim=128),
        text_encoder=MockLLMTextEncoder(vocab_size=128, hidden_dim=64, output_dim=128),
        state_encoder=StateEncoder(input_dim=2, output_dim=128),
        fusion=FusionMLP(input_dim=384, hidden_dim=256, output_dim=128),
        action_head=ActionHead(input_dim=128, action_dim=2),
    )


class TestMiniVLAForward:
    """Forward pass shape verification."""

    def test_output_shape_batch_4(self):
        model = _build_debug_mini_vla()
        batch = {
            "image": torch.randn(4, 3, 64, 64),
            "input_ids": torch.randint(0, 128, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "state": torch.randn(4, 2),
        }
        out = model(batch)
        assert out.shape == (4, 2)
        assert out.dtype == torch.float32

    def test_output_shape_batch_1(self):
        model = _build_debug_mini_vla()
        batch = {
            "image": torch.randn(1, 3, 64, 64),
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
            "state": torch.randn(1, 2),
        }
        out = model(batch)
        assert out.shape == (1, 2)

    def test_forward_without_attention_mask(self):
        model = _build_debug_mini_vla()
        batch = {
            "image": torch.randn(2, 3, 64, 64),
            "input_ids": torch.randint(0, 128, (2, 10)),
            "state": torch.randn(2, 2),
        }
        out = model(batch)
        assert out.shape == (2, 2)

    def test_forward_ignores_action_key(self):
        model = _build_debug_mini_vla()
        batch = {
            "image": torch.randn(4, 3, 64, 64),
            "input_ids": torch.randint(0, 128, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "state": torch.randn(4, 2),
            "action": torch.randn(4, 2),  # should be ignored
        }
        out = model(batch)
        assert out.shape == (4, 2)
