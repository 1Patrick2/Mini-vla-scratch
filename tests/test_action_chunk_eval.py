"""Tests for ActionChunk evaluation metrics."""

from __future__ import annotations

import torch

from mini_vla.evaluation.action_metrics import finite_ratio, mae, rmse


class TestChunkMetrics:
    def test_chunk_mae_all(self):
        pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        val = mae(pred, target)
        assert val >= 0

    def test_chunk_mae_first(self):
        pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        val = mae(pred[:1], target[:1])
        assert val >= 0

    def test_finite_ratio(self):
        pred = torch.randn(4, 2)
        assert finite_ratio(pred) == 1.0

    def test_rmse(self):
        pred = torch.randn(4, 2)
        target = torch.randn(4, 2)
        val = rmse(pred, target)
        assert val >= 0
