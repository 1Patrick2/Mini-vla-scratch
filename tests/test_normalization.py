"""Tests for action/state normalisation."""

import torch

from mini_vla.datasets.normalization import (
    ActionNormalizer,
    NormalizationStats,
    compute_stats,
    load_stats,
    save_stats,
)


class TestNormalizationStats:
    def test_create(self):
        stats = NormalizationStats(
            state_mean=torch.tensor([0.0, 0.0]),
            state_std=torch.tensor([1.0, 1.0]),
            action_mean=torch.tensor([0.5, -0.3]),
            action_std=torch.tensor([0.1, 0.2]),
        )
        assert stats.state_mean.shape == (2,)
        assert stats.action_mean[0] == 0.5


class TestActionNormalizer:
    def test_normalize_action(self):
        stats = NormalizationStats(
            state_mean=torch.zeros(2),
            state_std=torch.ones(2),
            action_mean=torch.tensor([1.0, 2.0]),
            action_std=torch.tensor([2.0, 4.0]),
        )
        norm = ActionNormalizer(stats)
        action = torch.tensor([3.0, 10.0])
        normalized = norm.normalize_action(action)
        # (3-1)/2 = 1.0, (10-2)/4 = 2.0
        assert torch.allclose(normalized, torch.tensor([1.0, 2.0]))

    def test_denormalize_action(self):
        stats = NormalizationStats(
            state_mean=torch.zeros(2),
            state_std=torch.ones(2),
            action_mean=torch.tensor([1.0, 2.0]),
            action_std=torch.tensor([2.0, 4.0]),
        )
        norm = ActionNormalizer(stats)
        normalized = torch.tensor([1.0, 2.0])
        denormalized = norm.denormalize_action(normalized)
        # 1*2+1 = 3.0, 2*4+2 = 10.0
        assert torch.allclose(denormalized, torch.tensor([3.0, 10.0]))

    def test_round_trip(self):
        """Normalise then denormalise should recover the original action."""
        stats = NormalizationStats(
            state_mean=torch.zeros(2),
            state_std=torch.ones(2),
            action_mean=torch.tensor([0.5, -0.3]),
            action_std=torch.tensor([1.2, 0.8]),
        )
        norm = ActionNormalizer(stats)
        original = torch.tensor([2.3, -1.7])
        recovered = norm.denormalize_action(norm.normalize_action(original))
        assert torch.allclose(original, recovered, atol=1e-6)

    def test_normalize_state(self):
        stats = NormalizationStats(
            state_mean=torch.tensor([1.0, 2.0]),
            state_std=torch.tensor([2.0, 3.0]),
            action_mean=torch.zeros(2),
            action_std=torch.ones(2),
        )
        norm = ActionNormalizer(stats)
        state = torch.tensor([3.0, 8.0])
        normalized = norm.normalize_state(state)
        # (3-1)/2 = 1.0, (8-2)/3 = 2.0
        assert torch.allclose(normalized, torch.tensor([1.0, 2.0]))

    def test_eps_guards_zero_std(self):
        stats = NormalizationStats(
            state_mean=torch.zeros(2),
            state_std=torch.ones(2),
            action_mean=torch.zeros(2),
            action_std=torch.zeros(2),  # zero std!
        )
        norm = ActionNormalizer(stats)
        action = torch.tensor([5.0, 5.0])
        # Should not divide by zero
        normalized = norm.normalize_action(action)
        assert torch.isfinite(normalized).all()


class TestComputeStats:
    def test_basic_stats(self):
        samples = [
            {"state": torch.tensor([0.0, 0.0]), "action": torch.tensor([1.0, 2.0])},
            {"state": torch.tensor([2.0, 4.0]), "action": torch.tensor([3.0, 6.0])},
        ]
        stats = compute_stats(samples, state_key="state", action_key="action")
        assert torch.allclose(stats.state_mean, torch.tensor([1.0, 2.0]))
        assert torch.allclose(stats.action_mean, torch.tensor([2.0, 4.0]))


class TestSaveLoadStats:
    def test_save_and_load(self, tmp_path):
        stats = NormalizationStats(
            state_mean=torch.tensor([0.0, 0.0]),
            state_std=torch.tensor([1.0, 1.0]),
            action_mean=torch.tensor([0.5, -0.3]),
            action_std=torch.tensor([0.1, 0.2]),
        )
        path = save_stats(stats, tmp_path / "stats.json")
        loaded = load_stats(path)
        assert torch.allclose(loaded.state_mean, stats.state_mean)
        assert torch.allclose(loaded.action_mean, stats.action_mean)
        assert torch.allclose(loaded.action_std, stats.action_std)
        assert loaded.eps == stats.eps
