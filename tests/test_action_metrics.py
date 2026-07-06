"""Tests for action evaluation metrics."""

import torch

from mini_vla.evaluation.action_metrics import (
    clip_rate,
    cosine_similarity,
    finite_ratio,
    mae,
    mse,
    per_dim_mae,
    rmse,
)


class TestMae:
    def test_identical(self):
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert mae(pred, pred) == 0.0

    def test_value(self):
        pred = torch.zeros(2, 2)
        target = torch.ones(2, 2)
        assert mae(pred, target) == 1.0


class TestMse:
    def test_identical(self):
        pred = torch.tensor([[1.0, 2.0]])
        assert mse(pred, pred) == 0.0


class TestRmse:
    def test_value(self):
        pred = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[3.0, 4.0]])
        # MSE = (9 + 16) / 2 = 12.5, RMSE = sqrt(12.5) ≈ 3.536
        assert abs(rmse(pred, target) - 3.536) < 0.01


class TestPerDimMae:
    def test_values(self):
        pred = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dims = per_dim_mae(pred, target)
        assert len(dims) == 2
        assert abs(dims[0] - 2.0) < 1e-6
        assert abs(dims[1] - 3.0) < 1e-6


class TestCosineSimilarity:
    def test_identical(self):
        pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert abs(cosine_similarity(pred, pred) - 1.0) < 1e-6

    def test_orthogonal(self):
        pred = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0]])
        assert abs(cosine_similarity(pred, target)) < 1e-6


class TestFiniteRatio:
    def test_all_finite(self):
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert finite_ratio(pred) == 1.0

    def test_with_nan(self):
        pred = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
        assert finite_ratio(pred) == 0.5


class TestClipRate:
    def test_no_clip(self):
        raw = torch.tensor([[0.01, 0.02]])
        clipped = raw.clone()
        assert clip_rate(raw, clipped) == 0.0

    def test_all_clipped(self):
        raw = torch.tensor([[0.1, 0.2]])
        clipped = torch.tensor([[0.05, 0.05]])
        assert clip_rate(raw, clipped) == 1.0
