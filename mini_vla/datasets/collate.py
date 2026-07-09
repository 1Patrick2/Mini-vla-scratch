"""Batch collation for Toy 2D dataset.

Takes a list of ``Toy2DDataset`` samples and produces a batched dict
ready for model training.

Supports automatic passthrough of additional fields (``action_chunk``,
``prev_action``, ``prev_state``, metadata, etc.) beyond the standard
five keys.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_toy_2d(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of Toy 2D samples into a training batch.

    Standard keys (``image``, ``input_ids``, ``attention_mask``, ``state``,
    ``action``) are always present with their original behaviour.

    Additional keys present in the batch samples are automatically passed
    through: tensor fields with matching shapes are stacked, scalars become
    tensors, and non-tensor metadata is preserved as a list.

    Args:
        batch: List of dicts from ``Toy2DDataset.__getitem__`` or
            a transform-wrapped dataset.

    Returns:
        Batched dict ready for model training.
    """
    # ── Standard collation (unchanged) ─────────────────────────────────
    images = torch.stack([s["image"] for s in batch])          # [B, 3, H, W]
    states = torch.stack([s["state"] for s in batch])          # [B, state_dim]
    actions = torch.stack([s["action"] for s in batch])        # [B, action_dim]

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
            padded_mask[i, : len(ids)] = 1

    result: Dict[str, Any] = {
        "image": images,
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "state": states,
        "action": actions,
    }

    # ── Automatic passthrough of additional fields ─────────────────────
    all_keys = set()
    for s in batch:
        all_keys.update(s.keys())
    extra_keys = all_keys - {"image", "input_ids", "attention_mask", "state", "action"}

    for key in extra_keys:
        values = [s.get(key) for s in batch]
        # Skip if no sample has this key
        if all(v is None for v in values):
            continue

        # Try stacking tensors with matching shape
        if all(isinstance(v, torch.Tensor) and v.shape == values[0].shape for v in values):  # noqa: E501
            result[key] = torch.stack(values)
        elif all(isinstance(v, torch.Tensor) and v.ndim == 0 for v in values):
            # Zero-dim scalars
            result[key] = torch.stack(values)
        elif all(isinstance(v, (int, float)) for v in values):
            result[key] = torch.tensor(values)
        elif all(isinstance(v, bool) for v in values):
            result[key] = torch.tensor(values, dtype=torch.bool)
        else:
            # Mixed types / strings / variable-shape tensors → list
            result[key] = values

    return result


__all__ = [
    "collate_toy_2d",
]
