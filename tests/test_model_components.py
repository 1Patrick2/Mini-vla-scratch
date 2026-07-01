"""Tests for StateEncoder, FusionMLP, and ActionHead."""

import torch

from mini_vla.models.action_head import ActionHead
from mini_vla.models.fusion import FusionMLP
from mini_vla.models.state_encoder import StateEncoder


class TestStateEncoder:
    """StateEncoder shape verification."""

    def test_output_shape(self):
        encoder = StateEncoder(input_dim=2, hidden_dim=64, output_dim=128)
        state = torch.randn(4, 2)
        out = encoder(state)
        assert out.shape == (4, 128)
        assert out.dtype == torch.float32

    def test_batch_1(self):
        encoder = StateEncoder(output_dim=128)
        state = torch.randn(1, 2)
        out = encoder(state)
        assert out.shape == (1, 128)


class TestFusionMLP:
    """FusionMLP shape verification."""

    def test_output_shape(self):
        fusion = FusionMLP(input_dim=384, hidden_dim=256, output_dim=128)
        image_feat = torch.randn(4, 128)
        text_feat = torch.randn(4, 128)
        state_feat = torch.randn(4, 128)
        out = fusion(image_feat, text_feat, state_feat)
        assert out.shape == (4, 128)
        assert out.dtype == torch.float32

    def test_batch_1(self):
        fusion = FusionMLP(output_dim=128)
        feats = [torch.randn(1, 128) for _ in range(3)]
        out = fusion(*feats)
        assert out.shape == (1, 128)


class TestActionHead:
    """ActionHead shape verification."""

    def test_output_shape(self):
        head = ActionHead(input_dim=128, hidden_dim=128, action_dim=2)
        fused = torch.randn(4, 128)
        out = head(fused)
        assert out.shape == (4, 2)
        assert out.dtype == torch.float32

    def test_batch_1(self):
        head = ActionHead(action_dim=2)
        fused = torch.randn(1, 128)
        out = head(fused)
        assert out.shape == (1, 2)

    def test_custom_action_dim(self):
        head = ActionHead(action_dim=7)
        fused = torch.randn(4, 128)
        out = head(fused)
        assert out.shape == (4, 7)
