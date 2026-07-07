"""Tests for the offline evaluator."""

import numpy as np
import torch

from mini_vla.evaluation.evaluator import evaluate_policy_on_dataset


class _DummyPolicy:
    """A policy that always predicts the mean of the target action."""

    def __init__(self, shift: float = 0.0):
        self.shift = shift

    def predict(self, sample):
        gt = sample["action"]
        return gt + self.shift


class _RandomPolicy:
    """A policy that predicts random actions."""

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def predict(self, sample):
        return torch.tensor(self.rng.randn(2).astype(np.float32))


def _make_samples(n=10, seed=42):
    rng = np.random.RandomState(seed)
    return [
        {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": rng.randn(2).astype(np.float32),
            "action": torch.tensor(rng.randn(2).astype(np.float32)),
            "episode_index": i // 5 if i > 0 else 0,
            "frame_index": i,
        }
        for i in range(n)
    ]


class TestEvaluatePolicyOnDataset:
    def test_report_contains_model_metrics(self):
        samples = _make_samples()
        policy = _DummyPolicy(shift=0.0)
        report = evaluate_policy_on_dataset(policy, samples, action_dim=2)

        assert "model" in report
        assert report["model"]["mae"] == 0.0
        assert report["model"]["mse"] == 0.0
        assert report["model"]["finite_ratio"] == 1.0
        assert report["num_samples"] == 10

    def test_report_contains_baselines(self):
        samples = _make_samples()
        policy = _RandomPolicy(seed=1)
        report = evaluate_policy_on_dataset(policy, samples, action_dim=2)

        assert "baselines" in report
        assert "zero_action" in report["baselines"]
        assert "mean_action" in report["baselines"]
        assert "previous_action" in report["baselines"]

    def test_perfect_policy_beats_baselines(self):
        """A perfect policy (shift=0) should outperform all baselines."""
        samples = _make_samples()
        policy = _DummyPolicy(shift=0.0)
        report = evaluate_policy_on_dataset(policy, samples, action_dim=2)

        model_mae = report["model"]["mae"]
        for name, bm in report["baselines"].items():
            if not isinstance(bm, dict) or "mae" not in bm:
                continue
            assert model_mae <= bm["mae"], (
                f"Perfect policy MAE {model_mae} > {name} MAE {bm['mae']}"
            )

    def test_max_samples_limits(self):
        samples = _make_samples(n=100)
        policy = _RandomPolicy()
        report = evaluate_policy_on_dataset(policy, samples, action_dim=2, max_samples=5)
        assert report["num_samples"] == 5
