"""Offline evaluator — compares a trained policy against baselines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from mini_vla.evaluation.action_metrics import (
    cosine_similarity,
    finite_ratio,
    mae,
    mse,
    per_dim_mae,
    rmse,
)
from mini_vla.evaluation.baselines import (
    MeanActionBaseline,
    PreviousActionBaseline,
    ZeroActionBaseline,
    compute_mean_action,
)


def evaluate_policy_on_dataset(
    policy: Any,
    dataset: Sequence[Dict[str, Any]],
    action_dim: int = 2,
    max_samples: int = 0,
    include_baselines: bool = True,
    baseline_dataset: Optional[Sequence[Dict[str, Any]]] = None,
    mean_action: Optional[torch.Tensor] = None,
    sort_by_episode_frame: bool = True,
) -> Dict[str, Any]:
    """Evaluate a policy on a dataset and compare against baselines.

    Args:
        policy: Object with ``predict(sample) -> Tensor[action_dim]``.
        dataset: Sequence of sample dicts containing ``action`` key.
        action_dim: Dimensionality of the action space.
        max_samples: If > 0, limit evaluation to this many samples.
        include_baselines: Whether to compute and include baseline metrics.
        baseline_dataset: Optional separate dataset for computing mean baseline
            (e.g. training set), to avoid data leakage.
        mean_action: Pre-computed mean action tensor.  Takes precedence over
            ``baseline_dataset``.
        sort_by_episode_frame: If True, sort pool by ``episode_index`` and
            ``frame_index`` before evaluation, so that
            ``PreviousActionBaseline`` behaves correctly.

    Returns:
        A serialisable report dict with keys:
        - num_samples
        - action_dim
        - model: dict of metric values
        - baselines (optional): dict of baseline_name -> metric values
        - per_dim_mae: list of per-dimension MAE for the model
        - mean_action_source: string describing where mean came from
    """
    # Build pool via explicit index iteration (avoids slicing bug with
    # adapters that don't support ``__getitem__`` with slices).
    n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    pool = [dataset[i] for i in range(n)]

    if not pool:
        return {"num_samples": 0, "action_dim": action_dim}

    # Sort by (episode_index, frame_index) so PreviousActionBaseline
    # behaves correctly
    if sort_by_episode_frame:
        pool = sorted(
            pool,
            key=lambda s: (
                s.get("episode_index", -1),
                s.get("frame_index", -1),
            ),
        )

    # Collect model predictions
    model_preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for sample in pool:
        pred = policy.predict(sample)
        model_preds.append(pred.cpu().clone().detach())
        gt = sample["action"]
        if not isinstance(gt, torch.Tensor):
            gt = torch.as_tensor(gt)
        targets.append(gt.cpu().clone().detach())

    model_preds_t = torch.stack(model_preds) if model_preds else torch.zeros(0, action_dim)
    targets_t = torch.stack(targets) if targets else torch.zeros(0, action_dim)

    report: Dict[str, Any] = {
        "num_samples": len(pool),
        "action_dim": action_dim,
    }

    # Model metrics
    report["model"] = _compute_metrics(model_preds_t, targets_t)
    report["per_dim_mae"] = per_dim_mae(model_preds_t, targets_t) if len(pool) > 0 else []

    # Baselines
    if include_baselines and len(pool) > 0:
        baselines_report: Dict[str, Any] = {}

        # ZeroAction
        zero = ZeroActionBaseline(action_dim)
        zero_preds = torch.stack([zero.predict(s) for s in pool])
        baselines_report["zero_action"] = _compute_metrics(zero_preds, targets_t)

        # MeanAction — avoid data leakage
        _mean_action_source: str
        if mean_action is not None:
            _mean_act = mean_action
            _mean_action_source = "provided"
        elif baseline_dataset is not None:
            _mean_act = compute_mean_action(
                [baseline_dataset[i] for i in range(min(len(baseline_dataset), 5000))]
            )
            _mean_action_source = "baseline_dataset"
        else:
            _mean_act = compute_mean_action(pool)
            _mean_action_source = "eval_pool"
        mean = MeanActionBaseline(_mean_act)
        mean_preds = torch.stack([mean.predict() for _ in pool])
        baselines_report["mean_action"] = _compute_metrics(mean_preds, targets_t)
        baselines_report["mean_action_source"] = _mean_action_source
        report["mean_action_source"] = _mean_action_source

        # PreviousAction
        prev = PreviousActionBaseline(action_dim)
        prev_preds = torch.stack([prev.predict(s) for s in pool])
        baselines_report["previous_action"] = _compute_metrics(prev_preds, targets_t)

        report["baselines"] = baselines_report

    return report


def _compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute a standard set of metrics between prediction and target tensors."""
    return {
        "mae": mae(preds, targets),
        "mse": mse(preds, targets),
        "rmse": rmse(preds, targets),
        "cosine_similarity": cosine_similarity(preds, targets),
        "finite_ratio": finite_ratio(preds),
        "clip_rate": None,  # reserved — raw vs clipped comparison not yet wired
    }
