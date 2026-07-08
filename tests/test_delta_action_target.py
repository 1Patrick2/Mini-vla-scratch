"""Tests for DeltaActionTargetWrapper."""

import torch

from mini_vla.datasets.transforms import DeltaActionTargetWrapper


class TestDeltaActionTarget:
    def test_raw_delta_correct(self):
        samples = [{
            "action_raw": torch.tensor([13.0, 8.0]),
            "action": torch.tensor([13.0, 8.0]),
            "prev_action_raw": torch.tensor([10.0, 10.0]),
            "prev_action": torch.tensor([10.0, 10.0]),
        }]
        wrapper = DeltaActionTargetWrapper(samples)
        s = wrapper[0]
        assert s["target_type"] == "delta_action"
        assert torch.allclose(s["delta_action_raw"], torch.tensor([3.0, -2.0]))
        assert torch.allclose(s["target_action_raw"], torch.tensor([13.0, 8.0]))
        # Training target should be delta_raw (no normalized available)
        assert torch.allclose(s["action"], torch.tensor([3.0, -2.0]))

    def test_normalized_delta_correct(self):
        samples = [{
            "action_raw": torch.tensor([13.0, 8.0]),
            "action_normalized": torch.tensor([0.4, -0.1]),
            "action": torch.tensor([0.4, -0.1]),
            "prev_action_raw": torch.tensor([10.0, 10.0]),
            "prev_action_normalized": torch.tensor([0.1, 0.2]),
            "prev_action": torch.tensor([0.1, 0.2]),
        }]
        wrapper = DeltaActionTargetWrapper(samples)
        s = wrapper[0]
        assert torch.allclose(s["delta_action_normalized"], torch.tensor([0.3, -0.3]))
        assert torch.allclose(s["target_action_normalized"], torch.tensor([0.4, -0.1]))
        # Training target should be normalized delta
        assert torch.allclose(s["action"], torch.tensor([0.3, -0.3]))

    def test_first_frame_delta(self):
        """First frame with prev_action=zero -> delta = action."""
        samples = [{
            "action_raw": torch.tensor([5.0, 5.0]),
            "action": torch.tensor([5.0, 5.0]),
            "prev_action_raw": torch.zeros(2),
            "prev_action": torch.zeros(2),
        }]
        wrapper = DeltaActionTargetWrapper(samples)
        s = wrapper[0]
        assert torch.allclose(s["delta_action_raw"], torch.tensor([5.0, 5.0]))
        assert torch.allclose(s["action"], torch.tensor([5.0, 5.0]))

    def test_target_raw_preserved(self):
        """target_action_raw should keep the original action_raw."""
        samples = [{
            "action_raw": torch.tensor([100.0, 200.0]),
            "action": torch.tensor([100.0, 200.0]),
            "prev_action_raw": torch.tensor([10.0, 20.0]),
            "prev_action": torch.tensor([10.0, 20.0]),
        }]
        wrapper = DeltaActionTargetWrapper(samples)
        s = wrapper[0]
        assert torch.allclose(s["target_action_raw"], torch.tensor([100.0, 200.0]))
        # Training target is delta
        assert torch.allclose(s["action"], torch.tensor([90.0, 180.0]))
