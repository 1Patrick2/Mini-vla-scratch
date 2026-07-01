"""Tests for vision encoder."""

import torch

from mini_vla.models.vision_encoder import SmallCNNVisionEncoder


class TestSmallCNNVisionEncoder:
    """Shape and dtype verification."""

    def test_output_shape_batch_4(self):
        encoder = SmallCNNVisionEncoder(output_dim=128)
        image = torch.randn(4, 3, 64, 64)
        out = encoder(image)
        assert out.shape == (4, 128)
        assert out.dtype == torch.float32

    def test_output_shape_batch_1(self):
        encoder = SmallCNNVisionEncoder(output_dim=128)
        image = torch.randn(1, 3, 64, 64)
        out = encoder(image)
        assert out.shape == (1, 128)

    def test_output_shape_batch_2(self):
        encoder = SmallCNNVisionEncoder(output_dim=128)
        image = torch.randn(2, 3, 64, 64)
        out = encoder(image)
        assert out.shape == (2, 128)

    def test_custom_output_dim(self):
        encoder = SmallCNNVisionEncoder(output_dim=256)
        image = torch.randn(2, 3, 64, 64)
        out = encoder(image)
        assert out.shape == (2, 256)
