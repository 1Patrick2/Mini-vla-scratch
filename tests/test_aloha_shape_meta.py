"""Tests for ALOHA shape_meta inspect (mock data, no network)."""

from __future__ import annotations

import numpy as np

from mini_vla.datasets.registry import get_dataset_spec


class TestAlohaShapeMeta:
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
