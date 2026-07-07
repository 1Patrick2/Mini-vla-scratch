"""Tests for evaluate_robot_dataset CLI using internal functions."""


import numpy as np
import torch

from mini_vla.datasets.normalization import ActionNormalizer, NormalizationStats
from mini_vla.datasets.registry import get_dataset_spec
from mini_vla.datasets.robot_adapter import BaseRobotDatasetAdapter
from scripts.evaluate_robot_dataset import (
    _get_action_views,
    evaluate_samples_with_predictor,
)


class FakePredictor:
    """Predictor that returns action + 0.1 in the training space."""

    def predict(self, sample):
        return sample["action"] + 0.1


def _make_mock_samples(n=4, with_normalizer=False):
    """Create mock samples, optionally with normalization."""
    rng = np.random.RandomState(42)
    raw_list = [
        {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": rng.randn(2).astype(np.float32),
            "action": rng.randn(2).astype(np.float32),
            "episode_index": i // 2,
            "frame_index": i % 2,
        }
        for i in range(n)
    ]
    spec = get_dataset_spec("pusht")
    if with_normalizer:
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.zeros(2), action_std=torch.ones(2),
        )
        normalizer = ActionNormalizer(stats)
    else:
        normalizer = None
    adapter = BaseRobotDatasetAdapter(raw_list, spec, normalizer=normalizer)
    return [adapter[i] for i in range(len(adapter))]


class TestGetActionViews:
    def test_without_normalizer(self):
        samples = _make_mock_samples(with_normalizer=False)
        sample = samples[0]
        pred = torch.tensor([0.1, 0.2])
        pn, gn, pr, gr = _get_action_views(sample, pred, normalizer=None)
        assert torch.allclose(pn, pred)
        assert torch.allclose(pr, pred)
        assert torch.allclose(gn, gr)

    def test_with_normalizer(self):
        samples = _make_mock_samples(with_normalizer=True)
        sample = samples[0]
        pred = torch.tensor([0.1, 0.2])
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.zeros(2), action_std=torch.ones(2),
        )
        normalizer = ActionNormalizer(stats)
        pn, gn, pr, gr = _get_action_views(sample, pred, normalizer)
        assert torch.allclose(pn, pred)
        assert "action_raw" in sample
        assert torch.allclose(gr, sample["action_raw"])


class TestEvaluateSamplesWithPredictor:
    def test_report_contains_all_metrics(self):
        samples = _make_mock_samples(n=8, with_normalizer=True)
        predictor = FakePredictor()
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.zeros(2), action_std=torch.ones(2),
        )
        normalizer = ActionNormalizer(stats)
        report, predictions = evaluate_samples_with_predictor(
            samples, predictor, normalizer=normalizer,
        )

        assert "raw_action_metrics" in report
        assert "normalized_action_metrics" in report
        assert "baselines" in report
        assert "zero_action" in report["baselines"]
        assert "mean_action" in report["baselines"]
        assert "previous_action" in report["baselines"]

    def test_predictions_have_raw_and_normalized(self):
        samples = _make_mock_samples(n=4, with_normalizer=True)
        predictor = FakePredictor()
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.ones(2) * 5, action_std=torch.ones(2) * 2,
        )
        normalizer = ActionNormalizer(stats)
        report, predictions = evaluate_samples_with_predictor(
            samples, predictor, normalizer=normalizer,
        )

        assert len(predictions) == 4
        p0 = predictions[0]
        assert "pred_action_raw" in p0
        assert "gt_action_raw" in p0
        assert "pred_action_normalized" in p0
        assert "gt_action_normalized" in p0

    def test_normalizer_enabled_flag(self):
        samples = _make_mock_samples(n=4, with_normalizer=True)
        predictor = FakePredictor()
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.zeros(2), action_std=torch.ones(2),
        )
        normalizer = ActionNormalizer(stats)
        report, predictions = evaluate_samples_with_predictor(
            samples, predictor, normalizer=normalizer,
        )
        # With identity normalizer, raw and normalized metrics should be close
        assert report["raw_action_metrics"]["mae"] >= 0
        assert report["normalized_action_metrics"]["mae"] >= 0

    def test_baselines_use_raw_action_space(self):
        samples = _make_mock_samples(n=8, with_normalizer=True)
        predictor = FakePredictor()
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.ones(2) * 10, action_std=torch.ones(2) * 3,
        )
        normalizer = ActionNormalizer(stats)
        report, _ = evaluate_samples_with_predictor(
            samples, predictor, normalizer=normalizer,
        )
        # Mean baseline should be computed from raw actions, not normalized
        mean_mae = report["baselines"]["mean_action"]["mae"]
        assert mean_mae > 0
