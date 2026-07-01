"""Tests for Toy 2D dataset and transforms."""

import numpy as np
import pytest

from mini_vla.datasets import Toy2DDataset, tokenize
from mini_vla.datasets.transforms import VOCAB


class TestTokenize:
    """Tokenisation logic — no data files needed."""

    def test_known_instruction(self):
        ids = tokenize("move red object to target")
        expected = [VOCAB["move"], VOCAB["red"], VOCAB["object"],
                    VOCAB["to"], VOCAB["target"]]
        assert ids == expected

    def test_padding(self):
        ids = tokenize("move", max_len=5)
        assert ids[0] == VOCAB["move"]
        assert ids[-1] == VOCAB["<pad>"]
        assert len(ids) == 5

    def test_unknown_word_becomes_unk(self):
        ids = tokenize("some unknown instruction")
        assert VOCAB["<unk>"] in ids


class TestToy2DDataset:
    """Dataset loading — requires generated toy data."""

    @pytest.fixture(autouse=True)
    def _generate_data(self, tmp_path):
        """Generate 2 small episodes for testing."""
        from scripts.generate_toy_data import generate_toy_data
        self.data_root = generate_toy_data(
            output_root=tmp_path,
            num_episodes=2,
            max_steps=4,
            image_size=64,
            seed=42,
        )

    def test_dataset_returns_correct_length(self):
        ds = Toy2DDataset(self.data_root)
        # 2 episodes × 4 steps = 8 samples
        assert len(ds) == 8

    def test_sample_has_expected_keys(self):
        ds = Toy2DDataset(self.data_root)
        sample = ds[0]
        assert "image" in sample
        assert "input_ids" in sample
        assert "state" in sample
        assert "action" in sample
        assert "instruction" in sample

    def test_image_shape(self):
        ds = Toy2DDataset(self.data_root)
        sample = ds[0]
        assert sample["image"].shape == (3, 64, 64)  # C, H, W

    def test_state_action_shape(self):
        ds = Toy2DDataset(self.data_root)
        sample = ds[0]
        assert sample["state"].shape == (2,)
        assert sample["action"].shape == (2,)

    def test_input_ids_dtype(self):
        ds = Toy2DDataset(self.data_root)
        sample = ds[0]
        assert sample["input_ids"].dtype == np.int64

    def test_instruction_string_preserved(self):
        ds = Toy2DDataset(self.data_root)
        sample = ds[0]
        assert isinstance(sample["instruction"], str)
        assert len(sample["instruction"]) > 0
