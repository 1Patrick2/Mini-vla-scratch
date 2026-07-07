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


def _make_samples_with_var_actions():
    """Return samples where train/eval have clearly distinct action means."""
    eval_s = [
        {"action": torch.tensor([10.0, 10.0]), "episode_index": 0, "frame_index": 0},
        {"action": torch.tensor([10.0, 10.0]), "episode_index": 0, "frame_index": 1},
    ]
    baseline_s = [
        {"action": torch.tensor([0.0, 0.0]), "episode_index": 1, "frame_index": 0},
        {"action": torch.tensor([0.0, 0.0]), "episode_index": 1, "frame_index": 1},
    ]
    return eval_s, baseline_s


class TestMeanActionBaselineSource:
    def test_baseline_dataset_source(self):
        eval_s, baseline_s = _make_samples_with_var_actions()
        policy = _DummyPolicy(shift=0.0)
        report = evaluate_policy_on_dataset(
            policy, eval_s, action_dim=2,
            baseline_dataset=baseline_s,
        )
        assert report["mean_action_source"] == "baseline_dataset"

    def test_baseline_avoids_eval_pool_leakage(self):
        """Mean baseline should be near [0,0] (baseline mean), not [10,10]."""
        eval_s, baseline_s = _make_samples_with_var_actions()
        policy = _DummyPolicy(shift=1.0)
        report = evaluate_policy_on_dataset(
            policy, eval_s, action_dim=2,
            baseline_dataset=baseline_s,
        )
        bm = report["baselines"]["mean_action"]
        # Mean computed from baseline_s is ~[0,0]; MAE with eval targets ~[10,10] = ~10
        assert bm["mae"] > 5.0, (
            f"Mean MAE {bm['mae']} should be high (baseline_dataset)"
        )


class TestPreviousActionSorting:
    def test_shuffled_input_still_correct(self):
        """Evaluator sorts by episode/frame, so PreviousActionBaseline works."""
        samples = [
            {"action": torch.tensor([5.0, 5.0]), "episode_index": 0, "frame_index": 1},
            {"action": torch.tensor([1.0, 1.0]), "episode_index": 0, "frame_index": 0},
            {"action": torch.tensor([9.0, 9.0]), "episode_index": 1, "frame_index": 0},
        ]
        policy = _DummyPolicy(shift=0.0)
        report = evaluate_policy_on_dataset(policy, samples, action_dim=2)
        bm = report["baselines"]["previous_action"]
        # After sorting: ep0_f0=[1,1] (no prev -> zero),
        #   ep0_f1=[5,5] (prev=[1,1]),
        #   ep1_f0=[9,9] (ep change -> zero)
        # MAE = mean(|1-0|,|5-1|,|9-0|)/2 = (1+4+9)/2 = 7
        assert bm["mae"] > 0, "Previous baseline should have non-zero error"
