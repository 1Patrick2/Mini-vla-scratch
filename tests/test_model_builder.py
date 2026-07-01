"""Tests for model builder, strict config validation, and DataLoader smoke."""

from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from mini_vla.datasets import Toy2DDataset
from mini_vla.datasets.collate import collate_toy_2d
from mini_vla.models.builder import build_model
from mini_vla.models.mini_vla import MiniVLA


class TestModelBuilderConfig:
    """Config-driven model construction — strict validation."""

    def test_build_from_real_yaml(self):
        config_path = Path("configs/model/mini_vla_cnn_llm.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model = build_model(config)
        assert isinstance(model, MiniVLA)

    def test_build_returns_mini_vla_instance(self):
        config = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 128},
                "text_encoder": {
                    "type": "mock_llm", "vocab_size": 128,
                    "hidden_dim": 64, "output_dim": 128,
                },
                "state_encoder": {"input_dim": 2, "output_dim": 128},
                "fusion": {"type": "concat_mlp", "input_dim": 384, "output_dim": 128},
                "action_head": {"input_dim": 128, "action_dim": 2},
            }
        }
        model = build_model(config)
        assert isinstance(model, MiniVLA)


class TestStrictTypeValidation:
    """Builder must reject unsupported component types."""

    def test_unsupported_vision_encoder_type_raises(self):
        config = {
            "model": {
                "vision_encoder": {"type": "resnet18"},
                "text_encoder": {"type": "mock_llm", "output_dim": 128},
                "state_encoder": {"output_dim": 128},
                "fusion": {"type": "concat_mlp"},
                "action_head": {"input_dim": 128},
            }
        }
        with pytest.raises(ValueError, match="vision_encoder.type"):
            build_model(config)

    def test_unsupported_text_encoder_type_raises(self):
        config = {
            "model": {
                "vision_encoder": {"type": "small_cnn"},
                "text_encoder": {"type": "gpt4"},
                "state_encoder": {"output_dim": 128},
                "fusion": {"type": "concat_mlp"},
                "action_head": {"input_dim": 128},
            }
        }
        with pytest.raises(ValueError, match="text_encoder.type"):
            build_model(config)

    def test_unsupported_fusion_type_raises(self):
        config = {
            "model": {
                "vision_encoder": {"type": "small_cnn"},
                "text_encoder": {"type": "mock_llm", "output_dim": 128},
                "state_encoder": {"output_dim": 128},
                "fusion": {"type": "cross_attention"},
                "action_head": {"input_dim": 128},
            }
        }
        with pytest.raises(ValueError, match="fusion.type"):
            build_model(config)


class TestFreezeTextEncoder:
    """Freeze behaviour for text_encoder."""

    def test_freeze_true_disables_grad(self):
        config = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 128},
                "text_encoder": {
                    "type": "mock_llm", "vocab_size": 128, "hidden_dim": 64,
                    "output_dim": 128, "freeze": True,
                },
                "state_encoder": {"output_dim": 128},
                "fusion": {"type": "concat_mlp"},
                "action_head": {"input_dim": 128},
            }
        }
        model = build_model(config)
        for param in model.text_encoder.parameters():
            assert not param.requires_grad, (
                f"Expected frozen text_encoder, but {param.shape} has requires_grad=True"
            )

    def test_freeze_false_keeps_grad(self):
        config = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"type": "small_cnn", "output_dim": 128},
                "text_encoder": {
                    "type": "mock_llm", "vocab_size": 128, "hidden_dim": 64,
                    "output_dim": 128, "freeze": False,
                },
                "state_encoder": {"output_dim": 128},
                "fusion": {"type": "concat_mlp"},
                "action_head": {"input_dim": 128},
            }
        }
        model = build_model(config)
        trainable = any(p.requires_grad for p in model.text_encoder.parameters())
        assert trainable


class TestCustomActionDim:
    """Model respects action_dim."""

    def test_action_dim_7(self):
        config = {
            "model": {
                "action_dim": 7,
                "vision_encoder": {"type": "small_cnn", "output_dim": 128},
                "text_encoder": {"type": "mock_llm", "output_dim": 128},
                "state_encoder": {"output_dim": 128},
                "fusion": {"type": "concat_mlp"},
                "action_head": {"input_dim": 128},
            }
        }
        model = build_model(config)
        batch = {
            "image": torch.randn(4, 3, 64, 64),
            "input_ids": torch.randint(0, 128, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "state": torch.randn(4, 2),
        }
        out = model(batch)
        assert out.shape == (4, 7)


class TestDataLoaderSmoke:
    """Real dataset batch → model forward (uses real YAML config)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from scripts.generate_toy_data import generate_toy_data
        self.data_root = generate_toy_data(
            output_root=tmp_path,
            num_episodes=2,
            max_steps=4,
            image_size=64,
            seed=42,
        )
        config_path = Path("configs/model/mini_vla_cnn_llm.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.model = build_model(yaml.safe_load(f))

    def test_dataloader_batch_passes_forward(self):
        ds = Toy2DDataset(self.data_root)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_toy_2d)
        batch = next(iter(loader))
        out = self.model(batch)
        assert out.shape == (4, 2)

    def test_dataloader_batch_batch_size_2(self):
        ds = Toy2DDataset(self.data_root)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_toy_2d)
        batch = next(iter(loader))
        out = self.model(batch)
        assert out.shape == (2, 2)
