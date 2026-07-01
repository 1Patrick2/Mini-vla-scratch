"""Tests for model builder and DataLoader-to-model smoke test."""

from pathlib import Path

import pytest
import yaml
from torch.utils.data import DataLoader

from mini_vla.datasets import Toy2DDataset
from mini_vla.datasets.collate import collate_toy_2d
from mini_vla.models.builder import build_model


class TestModelBuilder:
    """Config-driven model construction."""

    def test_build_from_yaml_config(self):
        config_path = Path("configs/model/mini_vla_cnn_llm.yaml")
        assert config_path.exists()
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model = build_model(config)
        assert model is not None

    def test_build_returns_mini_vla_instance(self):
        config = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"output_dim": 128},
                "text_encoder": {"vocab_size": 128, "hidden_dim": 64, "output_dim": 128},
                "state_encoder": {"input_dim": 2, "output_dim": 128},
                "fusion": {"input_dim": 384, "output_dim": 128},
                "action_head": {"input_dim": 128, "action_dim": 2},
            }
        }
        model = build_model(config)
        from mini_vla.models.mini_vla import MiniVLA
        assert isinstance(model, MiniVLA)


class TestDataLoaderSmoke:
    """Real dataset batch → model forward smoke test."""

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
        config = {
            "model": {
                "action_dim": 2,
                "vision_encoder": {"output_dim": 128},
                "text_encoder": {"vocab_size": 128, "hidden_dim": 64, "output_dim": 128},
                "state_encoder": {"input_dim": 2, "output_dim": 128},
                "fusion": {"input_dim": 384, "output_dim": 128},
                "action_head": {"input_dim": 128, "action_dim": 2},
            }
        }
        self.model = build_model(config)

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
