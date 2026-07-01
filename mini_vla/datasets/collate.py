"""Batch collation for Toy 2D dataset.

Takes a list of ``Toy2DDataset`` samples and produces a batched dict
ready for model training.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_toy_2d(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate a list of Toy 2D samples into a training batch.

    Args:
        batch: List of dicts from ``Toy2DDataset.__getitem__``.

    Returns:
        Batched dict with keys ``image``, ``input_ids``, ``attention_mask``,
        ``state``, ``action``. All values are ``torch.Tensor``.
    """
    images = torch.stack([s["image"] for s in batch])          # [B, 3, H, W]
    states = torch.stack([s["state"] for s in batch])          # [B, 2]
    actions = torch.stack([s["action"] for s in batch])        # [B, 2]

    # Pad input_ids and attention_mask to the maximum length in this batch
    ids_list = [s["input_ids"] for s in batch]
    mask_list = [s.get("attention_mask") for s in batch]
    max_len = max(len(ids) for ids in ids_list)
    padded_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, ids in enumerate(ids_list):
        padded_ids[i, : len(ids)] = ids
        if mask_list[i] is not None:
            padded_mask[i, : len(mask_list[i])] = mask_list[i]
        else:
            # If no attention_mask provided, default to all-ones
            padded_mask[i, : len(ids)] = 1

    return {
        "image": images,
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "state": states,
        "action": actions,
    }


__all__ = [
    "collate_toy_2d",
]
