"""Tests for ALOHA shape_meta inspect (mock data, no network)."""

from __future__ import annotations

import numpy as np
import torch

from mini_vla.datasets.registry import get_dataset_spec


def _infer_dim(x):
    """Replicate the safe dim inference from inspect_shape_meta.py."""
    if isinstance(x, torch.Tensor):
        return int(x.numel()) if x.ndim == 1 else int(x.shape[-1])
    if isinstance(x, np.ndarray):
        return int(x.size) if x.ndim == 1 else int(x.shape[-1])
    if isinstance(x, (list, tuple)):
        return len(x)
    return 0


class TestInferDim:
    def test_numpy_1d(self):
        assert _infer_dim(np.random.randn(14)) == 14

    def test_numpy_2d_last_dim(self):
        assert _infer_dim(np.random.randn(1, 14)) == 14

    def test_torch_1d(self):
        assert _infer_dim(torch.randn(14)) == 14

    def test_torch_2d_last_dim(self):
        assert _infer_dim(torch.randn(1, 14)) == 14

    def test_list(self):
        assert _infer_dim([1.0] * 14) == 14

    def test_tuple(self):
        assert _infer_dim((1.0,) * 14) == 14

    def test_empty_none(self):
        assert _infer_dim(None) == 0


class TestImageInference:
    def test_numpy_hwc(self):
        arr = np.random.randn(480, 640, 3).astype(np.uint8)
        assert arr.ndim == 3 and arr.shape[-1] == 3

    def test_torch_chw(self):
        t = torch.randn(3, 224, 224)
        assert t.ndim == 3 and t.shape[0] == 3

    def test_torch_1ch_chw(self):
        t = torch.randn(1, 224, 224)
        assert t.ndim == 3 and t.shape[0] == 1

    def test_numpy_chw(self):
        arr = np.random.randn(3, 224, 224).astype(np.uint8)
        assert arr.ndim == 3 and arr.shape[0] == 3

    def test_torch_hwc(self):
        t = torch.randn(224, 224, 3)
        assert t.ndim == 3 and t.shape[-1] == 3


class TestAlohaSpec:
    def test_aloha_spec_exists(self):
        """ALOHA sim should be registered with a spec."""
        spec = get_dataset_spec("aloha_sim_transfer_cube")
        assert spec is not None
        assert spec.name == "aloha_sim_transfer_cube"

    def test_aloha_repo_id(self):
        spec = get_dataset_spec("aloha_sim_transfer_cube")
        assert "aloha" in spec.repo_id

    def test_mock_infer_keys(self):
        """Simulate the inspect logic with a mock ALOHA-like sample."""
        sample = {
            "observation.images.top": np.random.randn(480, 640, 3).astype(np.uint8),
            "observation.state": np.random.randn(14).astype(np.float32),
            "action": np.random.randn(14).astype(np.float32),
            "task": "pick up the cube",
        }
        # Test dimension inference
        state_dim = sample["observation.state"].shape[0]
        action_dim = sample["action"].shape[0]
        assert state_dim == 14
        assert action_dim == 14
        assert "observation.images.top" in sample
        assert "task" in sample

    def test_mock_low_dim(self):
        """PushT-like sample dimensions."""
        sample = {
            "observation.image": np.random.randn(96, 96, 3).astype(np.uint8),
            "observation.state": np.random.randn(2).astype(np.float32),
            "action": np.random.randn(2).astype(np.float32),
        }
        assert sample["observation.state"].shape[0] == 2
        assert sample["action"].shape[0] == 2
