"""Tests for the PushT dataset adapter."""

import numpy as np
import torch
from PIL import Image

from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter


def _make_mock_sample(image_hwc=False):
    """Return a mock PushT-like sample with HWC uint8 image by default."""
    rng = np.random.RandomState(42)
    if image_hwc:
        img = rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    else:
        img = rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    return {
        "observation.image": img,
        "observation.state": rng.randn(2).astype(np.float32),
        "action": rng.randn(2).astype(np.float32),
        "episode_index": 0,
        "frame_index": 0,
        "timestamp": 0.0,
        "next.reward": 0.0,
        "next.done": False,
        "next.success": False,
        "task_index": 0,
    }


class TestPushTDatasetAdapterConstruction:
    def test_adapter_accepts_list(self):
        samples = [_make_mock_sample() for _ in range(3)]
        adapter = PushTDatasetAdapter(samples)
        assert len(adapter) == 3


class TestPushTDatasetAdapterOutput:
    @classmethod
    def setup_class(cls):
        cls.samples = [_make_mock_sample() for _ in range(5)]
        cls.adapter = PushTDatasetAdapter(cls.samples)

    def test_sample_keys(self):
        sample = self.adapter[0]
        expected_keys = {
            "image", "input_ids", "attention_mask",
            "state", "action", "instruction",
        }
        assert expected_keys.issubset(sample.keys()), (
            f"Missing keys: {expected_keys - sample.keys()}"
        )

    def test_image_shape_and_dtype(self):
        sample = self.adapter[0]
        img = sample["image"]
        assert img.shape == (3, 64, 64), f"Expected (3,64,64), got {img.shape}"
        assert img.dtype == torch.float32
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_state_shape(self):
        sample = self.adapter[0]
        assert sample["state"].shape == (2,)
        assert sample["state"].dtype == torch.float32

    def test_action_shape(self):
        sample = self.adapter[0]
        assert sample["action"].shape == (2,)
        assert sample["action"].dtype == torch.float32

    def test_instruction(self):
        sample = self.adapter[0]
        assert sample["instruction"] == "push the T block to the target"

    def test_input_ids_present(self):
        sample = self.adapter[0]
        assert "input_ids" in sample
        assert sample["input_ids"].ndim == 1

    def test_attention_mask_present(self):
        sample = self.adapter[0]
        assert "attention_mask" in sample
        assert sample["attention_mask"].ndim == 1

    def test_metadata_fields(self):
        sample = self.adapter[0]
        assert "episode_index" in sample
        assert "frame_index" in sample
        assert "timestamp" in sample
        assert "next_reward" in sample
        assert "next_done" in sample
        assert "next_success" in sample


class TestPushTDatasetAdapterImageVariants:
    def test_hwc_uint8(self):
        rng = np.random.RandomState(1)
        sample = {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
        }
        adapter = PushTDatasetAdapter([sample])
        img = adapter[0]["image"]
        assert img.shape == (3, 64, 64)

    def test_pil_image(self):
        pil = Image.new("RGB", (96, 96), color=(128, 64, 32))
        sample = {
            "observation.image": pil,
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
        }
        adapter = PushTDatasetAdapter([sample])
        img = adapter[0]["image"]
        assert img.shape == (3, 64, 64)
