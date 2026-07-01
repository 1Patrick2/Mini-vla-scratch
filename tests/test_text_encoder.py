"""Tests for LLM-ready text encoder."""

import pytest
import torch

from mini_vla.models.language_encoder import MockLLMTextEncoder


class TestMockLLMTextEncoder:
    """MockLLMTextEncoder — no transformers dependency."""

    @pytest.fixture
    def encoder(self):
        return MockLLMTextEncoder(vocab_size=128, hidden_dim=64, output_dim=128)

    def test_output_shape(self, encoder):
        input_ids = torch.randint(0, 128, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        out = encoder(input_ids, attention_mask)
        assert out.shape == (4, 128)
        assert out.dtype == torch.float32

    def test_output_without_attention_mask(self, encoder):
        input_ids = torch.randint(0, 128, (2, 10))
        out = encoder(input_ids)
        assert out.shape == (2, 128)

    def test_padding_does_not_affect_shape(self, encoder):
        input_ids = torch.randint(0, 128, (4, 16))
        mask = torch.ones(4, 16, dtype=torch.long)
        # Set some padding
        mask[:, 8:] = 0
        input_ids[:, 8:] = 0
        out = encoder(input_ids, mask)
        assert out.shape == (4, 128)

    def test_all_padding_does_not_produce_nan(self, encoder):
        input_ids = torch.zeros(2, 10, dtype=torch.long)  # all pad
        mask = torch.zeros(2, 10, dtype=torch.long)
        out = encoder(input_ids, mask)
        assert out.shape == (2, 128)
        assert not torch.isnan(out).any()

    def test_custom_vocab_and_dim(self):
        encoder = MockLLMTextEncoder(vocab_size=256, hidden_dim=32, output_dim=64)
        input_ids = torch.randint(0, 256, (4, 16))
        out = encoder(input_ids)
        assert out.shape == (4, 64)


class TestLLMTextEncoder:
    """LLMTextEncoder — class existence only, no real AutoModel download."""

    def test_llm_text_encoder_class_is_importable(self):
        from mini_vla.models.language_encoder import LLMTextEncoder
        assert LLMTextEncoder is not None

    def test_llm_text_encoder_is_subclass(self):
        from mini_vla.models.language_encoder import (
            BaseTextEncoder,
            LLMTextEncoder,
        )
        assert issubclass(LLMTextEncoder, BaseTextEncoder)
