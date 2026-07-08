"""Tests for evaluate_robot_dataset with delta action."""

import torch

from mini_vla.datasets.normalization import ActionNormalizer, NormalizationStats
from scripts.evaluate_robot_dataset import _get_action_views


class TestEvaluateDeltaAction:
    def test_delta_raw_reconstruction(self):
        """pred_action_raw = prev_action_raw + pred_delta_raw."""
        sample = {
            "target_type": "delta_action",
            "prev_action_raw": torch.tensor([10.0, 10.0]),
            "target_action_raw": torch.tensor([12.0, 9.0]),
            "action": torch.tensor([12.0, 9.0]),
        }
        pred_delta = torch.tensor([2.0, -1.0])
        pn, gn, pr, gr = _get_action_views(sample, pred_delta, normalizer=None)
        # Reconstructed pred_action_raw = 10+2=12, 10+(-1)=9
        assert torch.allclose(pr, torch.tensor([12.0, 9.0]))
        assert torch.allclose(gr, torch.tensor([12.0, 9.0]))

    def test_delta_with_normalizer(self):
        """Delta + normalizer: denormalize after reconstruction."""
        stats = NormalizationStats(
            state_mean=torch.zeros(2), state_std=torch.ones(2),
            action_mean=torch.zeros(2), action_std=torch.ones(2),
        )
        normalizer = ActionNormalizer(stats)
        sample = {
            "target_type": "delta_action",
            "prev_action_normalized": torch.tensor([0.1, 0.2]),
            "target_action_normalized": torch.tensor([0.4, -0.1]),
            "action_normalized": torch.tensor([0.4, -0.1]),
            "action": torch.tensor([0.4, -0.1]),
            "target_action_raw": torch.tensor([12.0, 9.0]),
        }
        pred_delta_norm = torch.tensor([0.3, -0.3])
        pn, gn, pr, gr = _get_action_views(sample, pred_delta_norm, normalizer)
        # pred_norm = 0.1+0.3=0.4, 0.2+(-0.3)=-0.1
        assert torch.allclose(pn, torch.tensor([0.4, -0.1]))
        # gt_norm from target
        assert torch.allclose(gn, torch.tensor([0.4, -0.1]))
