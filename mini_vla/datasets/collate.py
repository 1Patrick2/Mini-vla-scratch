"""Batch collation for MiniVLA.

``collate_minivla_batch`` is the primary collation function.  It handles
Toy 2D, PushT, ALOHA, and robot-dataset samples with automatic
passthrough of additional fields (``action_chunk``, ``prev_action``,
``prev_state``, metadata, etc.).

``collate_toy_2d`` is retained as a backward-compatible alias.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_minivla_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of MiniVLA samples into a training batch.

    Standard keys (``image``, ``input_ids``, ``attention_mask``, ``state``,
    ``action``) are always present with their original behaviour.

    Additional keys present in the batch samples are automatically passed
    through: tensor fields with matching shapes are stacked, scalars become
    tensors, and non-tensor metadata is preserved as a list.

    This collation works for Toy 2D, PushT, and general robot datasets
    (ALOHA, LIBERO, etc.) because it does not assume a fixed key set.

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
            result[key] = torch.stack(values)
        elif all(isinstance(v, (int, float)) for v in values):
            result[key] = torch.tensor(values)
        elif all(isinstance(v, bool) for v in values):
            result[key] = torch.tensor(values, dtype=torch.bool)
        else:
            # Mixed types / strings / variable-shape tensors → list
            result[key] = values

    return result


def collate_toy_2d(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Backward-compatible alias for :func:`collate_minivla_batch`.

    Retained so that Stage 1–7 tests and scripts continue to work
    without changes.
    """
    return collate_minivla_batch(batch)


__all__ = [
    "collate_minivla_batch",
    "collate_toy_2d",
]
