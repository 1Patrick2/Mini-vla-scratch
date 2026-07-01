"""Batch collation for Toy 2D dataset.

Takes a list of ``Toy2DDataset`` samples and produces a batched dict
ready for model training.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def collate_toy_2d(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate a list of Toy 2D samples into a training batch.

    Args:
        batch: List of dicts from ``Toy2DDataset.__getitem__``.

    Returns:
        Batched dict with keys ``image``, ``input_ids``, ``state``, ``action``.
        All values are ``torch.Tensor``.
    """
    images = np.stack([s["image"] for s in batch])          # [B, 3, H, W]
    states = np.stack([s["state"] for s in batch])          # [B, 2]
    actions = np.stack([s["action"] for s in batch])        # [B, 2]

    # Pad input_ids to the maximum length in this batch
    ids_list = [s["input_ids"] for s in batch]
    max_len = max(len(ids) for ids in ids_list)
    padded = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = ids

    return {
        "image": torch.from_numpy(images),
        "input_ids": torch.from_numpy(padded),
        "state": torch.from_numpy(states),
        "action": torch.from_numpy(actions),
    }


__all__ = [
    "collate_toy_2d",
]
