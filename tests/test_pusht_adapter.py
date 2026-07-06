"""Tests for the PushT dataset adapter."""

import numpy as np
import torch
from PIL import Image

from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter, _get_by_key, _process_image


def _make_mock_sample():
    """Return a mock PushT-like sample with HWC uint8 image by default."""
    rng = np.random.RandomState(42)
    return {
        "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
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


class TestGetByKey:
    def test_flat_key(self):
        sample = {"observation.image": "val"}
        assert _get_by_key(sample, "observation.image") == "val"

    def test_nested_key(self):
        sample = {"observation": {"image": "val"}}
        assert _get_by_key(sample, "observation.image") == "val"

    def test_flat_preferred_over_nested(self):
        sample = {"observation.image": "flat", "observation": {"image": "nested"}}
        assert _get_by_key(sample, "observation.image") == "flat"

    def test_missing_key_raises(self):
        import pytest
        with pytest.raises(KeyError):
            _get_by_key({"a": 1}, "observation.image")


class TestProcessImage:
    def test_chw_torch_uint8(self):
        img = torch.randint(0, 256, (3, 96, 96), dtype=torch.uint8)
        result = _process_image(img)
        assert result.shape == (3, 64, 64)
        assert result.dtype == torch.float32
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_chw_torch_float(self):
        img = torch.rand(3, 96, 96, dtype=torch.float32)
        result = _process_image(img)
        assert result.shape == (3, 64, 64)
        assert result.dtype == torch.float32

    def test_hwc_float_preserves_values(self):
        """A mid-grey float image should stay mid-grey after processing."""
        img_np = np.full((96, 96, 3), 0.5, dtype=np.float32)
        result = _process_image(img_np)
        mean_val = result.mean().item()
        assert 0.45 < mean_val < 0.55, (
            f"Mean value {mean_val:.4f} should be ~0.5, not near-zero"
        )

    def test_chw_float_preserves_values(self):
        img = torch.full((3, 96, 96), 0.5, dtype=torch.float32)
        result = _process_image(img)
        mean_val = result.mean().item()
        assert 0.45 < mean_val < 0.55, (
            f"Mean value {mean_val:.4f} should be ~0.5, not near-zero"
        )

    def test_pil_preserves_rough_color(self):
        pil = Image.new("RGB", (96, 96), color=(128, 64, 32))
        result = _process_image(pil)
        # RGB (128,64,32) / 255 ≈ (0.502, 0.251, 0.125)
        r, g, b = result[:, 0, 0]
        assert abs(r - 0.502) < 0.02
        assert abs(g - 0.251) < 0.02

    def test_hwc_float_black(self):
        """All-zero float image should stay near zero, not become NaN."""
        img_np = np.zeros((96, 96, 3), dtype=np.float32)
        result = _process_image(img_np)
        assert result.mean() < 0.001

    def test_nested_raw(self):
        """Adapter accepts nested sample format (raw['observation']['image'])."""
        rng = np.random.RandomState(1)
        raw = {
            "observation": {
                "image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
                "state": rng.randn(2).astype(np.float32),
            },
            "action": rng.randn(2).astype(np.float32),
            "episode_index": 0,
        }
        adapter = PushTDatasetAdapter([raw])
        sample = adapter[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["state"].shape == (2,)
        assert sample["action"].shape == (2,)

    def test_flat_vs_nested_both_read(self):
        """Both flat and nested key formats should produce identical output shapes."""
        rng = np.random.RandomState(2)
        flat_raw = {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
        }
        rng = np.random.RandomState(2)
        nested_raw = {
            "observation": {
                "image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
                "state": np.zeros(2, dtype=np.float32),
            },
            "action": np.zeros(2, dtype=np.float32),
        }
        flat_adapter = PushTDatasetAdapter([flat_raw])
        nested_adapter = PushTDatasetAdapter([nested_raw])
        assert flat_adapter[0]["image"].shape == (3, 64, 64)
        assert nested_adapter[0]["image"].shape == (3, 64, 64)
        assert flat_adapter[0]["state"].shape == (2,)
        assert nested_adapter[0]["state"].shape == (2,)


class TestPushTDatasetAdapterConstruction:
    def test_adapter_accepts_list(self):
        samples = [_make_mock_sample() for _ in range(3)]
        adapter = PushTDatasetAdapter(samples)
        assert len(adapter) == 3

    def test_adapter_without_metadata(self):
        """Adapter works when optional metadata keys are absent."""
        raw = {
            "observation.image": np.zeros((96, 96, 3), dtype=np.uint8),
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
        }
        adapter = PushTDatasetAdapter([raw])
        sample = adapter[0]
        assert sample["image"].shape == (3, 64, 64)


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
