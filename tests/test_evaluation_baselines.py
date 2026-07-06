"""Tests for evaluation baselines."""

import torch

from mini_vla.evaluation.baselines import (
    MeanActionBaseline,
    PreviousActionBaseline,
    ZeroActionBaseline,
    compute_mean_action,
)


class TestZeroActionBaseline:
    def test_output_shape(self):
        baseline = ZeroActionBaseline(action_dim=2)
        action = baseline.predict(sample={})
        assert action.shape == (2,)
        assert action.sum() == 0.0

    def test_repr(self):
        baseline = ZeroActionBaseline(action_dim=2)
        assert "ZeroAction" in repr(baseline)


class TestMeanActionBaseline:
    def test_returns_mean(self):
        mean = torch.tensor([0.5, -0.3])
        baseline = MeanActionBaseline(mean)
        action = baseline.predict()
        assert torch.allclose(action, mean)

    def test_repr(self):
        baseline = MeanActionBaseline(torch.tensor([1.0, 2.0]))
        assert "MeanAction" in repr(baseline)


class TestPreviousActionBaseline:
    def test_first_frame_returns_zero(self):
        baseline = PreviousActionBaseline(action_dim=2)
        action = baseline.predict({"action": torch.tensor([0.1, 0.2])})
        # First frame should return zeros
        assert torch.allclose(action, torch.zeros(2))

    def test_second_frame_returns_previous(self):
        baseline = PreviousActionBaseline(action_dim=2)
        baseline.predict({"action": torch.tensor([0.1, 0.2])})
        action = baseline.predict({"action": torch.tensor([0.3, 0.4])})
        assert torch.allclose(action, torch.tensor([0.1, 0.2]))

    def test_episode_boundary_resets(self):
        baseline = PreviousActionBaseline(action_dim=2)
        # Episode 0, frame 1
        baseline.predict({"episode_index": 0, "action": torch.tensor([0.5, 0.6])})
        # Episode 1, frame 0 — should reset
        action = baseline.predict({"episode_index": 1, "action": torch.tensor([0.7, 0.8])})
        assert torch.allclose(action, torch.zeros(2))

    def test_reset_method(self):
        baseline = PreviousActionBaseline(action_dim=2)
        baseline.predict({"action": torch.tensor([0.1, 0.2])})
        baseline.reset()
        action = baseline.predict({"action": torch.tensor([0.3, 0.4])})
        assert torch.allclose(action, torch.zeros(2))

    def test_repr(self):
        baseline = PreviousActionBaseline(action_dim=2)
        assert "PreviousAction" in repr(baseline)


class TestComputeMeanAction:
    def test_mean_value(self):
        samples = [
            {"action": torch.tensor([1.0, 2.0])},
            {"action": torch.tensor([3.0, 4.0])},
        ]
        mean = compute_mean_action(samples)
        assert torch.allclose(mean, torch.tensor([2.0, 3.0]))

    def test_empty_returns_zeros(self):
        mean = compute_mean_action([])
        assert torch.allclose(mean, torch.zeros(2))
