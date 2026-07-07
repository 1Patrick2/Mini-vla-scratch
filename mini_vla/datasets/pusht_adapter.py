"""PushT dataset adapter — converts PushT samples to MiniVLA format.

Usage:
    adapter = PushTDatasetAdapter(raw_dataset)
    sample = adapter[0]  # MiniVLA-format sample dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from mini_vla.datasets.pusht_inspection import DEFAULT_PUSHT_KEYS
from mini_vla.datasets.transforms import build_attention_mask, tokenize

_IMAGE_KEY_CANDIDATES = [
    "observation.image",
    "observation.images.top",
    "observation.images.main",
    "image",
]


def _get_by_key(sample: Dict[str, Any], key: str) -> Any:
    """Get a value from a sample dict supporting both flat and nested keys.

    ``sample["observation.image"]`` (flat) is tried first;
    ``sample["observation"]["image"]`` (nested) is tried second.

    Raises:
        KeyError: If neither form exists.
    """
    if key in sample:
        return sample[key]
    parts = key.split(".", 1)
    if len(parts) == 2 and parts[0] in sample:
        child = sample[parts[0]]
        if isinstance(child, dict) and parts[1] in child:
            return child[parts[1]]
    raise KeyError(
        f"Key '{key}' not found in sample (tried flat and nested). "
        f"Available keys: {list(sample.keys())}"
    )


def _maybe_get_by_key(sample: Dict[str, Any], key: str) -> Any:
    """Like ``_get_by_key`` but returns ``None`` when the key is missing."""
    try:
        return _get_by_key(sample, key)
    except KeyError:
        return None


def _process_image(img_data: Any, image_size: int = 64) -> torch.Tensor:
    """Convert a PushT image to a CHW float32 tensor in [0, 1].

    Supports:
    - HWC numpy uint8  [H,W,3]  in [0, 255]
    - HWC numpy float  [H,W,3]  in [0, 1]
    - CHW torch.uint8  [3,H,W]  in [0, 255]
    - CHW torch.float  [3,H,W]  in [0, 1]
    - PIL Image

    Returns:
        Tensor[3, image_size, image_size], float32, range [0, 1].
    """
    # ── PIL ──────────────────────────────────────────────────────────
    if isinstance(img_data, Image.Image):
        pil = img_data

    # ── torch.Tensor ──────────────────────────────────────────────────
    elif isinstance(img_data, torch.Tensor):
        ndim = img_data.ndim
        if ndim == 3 and img_data.shape[0] in (1, 3):
            # CHW — already the target layout
            img = img_data.float()
            if img.max() > 1.0:
                img = img / 255.0
            img = img.clamp(0, 1)
            if img.shape[1] != image_size or img.shape[2] != image_size:
                # Resize via PIL
                pil = Image.fromarray((img.permute(1, 2, 0) * 255).byte().numpy())
                pil = pil.resize((image_size, image_size), Image.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                return torch.from_numpy(arr).permute(2, 0, 1).clamp(0, 1)
            return img
        # HWC tensor
        pil = Image.fromarray(img_data.byte().numpy())

    # ── numpy ─────────────────────────────────────────────────────────
    elif isinstance(img_data, np.ndarray):
        ndim = img_data.ndim
        if ndim == 3 and img_data.shape[-1] in (1, 3):
            # HWC
            arr = img_data.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = arr.clip(0, 1)
            pil = Image.fromarray((arr * 255).astype(np.uint8))
        elif ndim == 3 and img_data.shape[0] in (1, 3):
            # CHW
            arr = img_data.transpose(1, 2, 0).astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = arr.clip(0, 1)
            pil = Image.fromarray((arr * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unexpected image array shape: {img_data.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(img_data)}")

    # ── Resize and return ────────────────────────────────────────────
    if pil.size != (image_size, image_size):
        pil = pil.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    # (H, W, C) → (C, H, W)
    return torch.from_numpy(arr).permute(2, 0, 1).clamp(0, 1)


class PushTDatasetAdapter:
    """Adapt PushT-like samples to MiniVLA format.

    Args:
        base_dataset: A list, list-like, or sequence of PushT sample dicts.
        image_key: Key for the image in the PushT sample (default
            ``observation.image``).
        image_keys: Optional list of candidate image keys tried in order.
            If provided, takes precedence over ``image_key``.
        state_key: Key for the state.
        action_key: Key for the action.
        image_size: Target image size (default 64).
        max_text_len: Max token length for instruction (default 16).
        instruction: Fixed instruction string.  If ``None``, attempts to
            read from sample metadata.
        episode_index_key: Key for episode index.
        frame_index_key: Key for frame index.
        timestamp_key: Key for timestamp.
        reward_key: Key for reward.
        done_key: Key for done flag.
        success_key: Key for success flag.
    """

    def __init__(
        self,
        base_dataset: Sequence[Dict[str, Any]],
        image_key: str = DEFAULT_PUSHT_KEYS["image_key"],
        image_keys: Optional[List[str]] = None,
        state_key: str = DEFAULT_PUSHT_KEYS["state_key"],
        action_key: str = DEFAULT_PUSHT_KEYS["action_key"],
        image_size: int = 64,
        max_text_len: int = 16,
        instruction: Optional[str] = "push the T block to the target",
        episode_index_key: str = DEFAULT_PUSHT_KEYS["episode_index_key"],
        frame_index_key: str = DEFAULT_PUSHT_KEYS["frame_index_key"],
        timestamp_key: str = DEFAULT_PUSHT_KEYS["timestamp_key"],
        reward_key: str = DEFAULT_PUSHT_KEYS["reward_key"],
        done_key: str = DEFAULT_PUSHT_KEYS["done_key"],
        success_key: str = DEFAULT_PUSHT_KEYS["success_key"],
    ) -> None:
        self.base_dataset = base_dataset
        self.image_key = image_key
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.instruction = instruction
        self.episode_index_key = episode_index_key
        self.frame_index_key = frame_index_key
        self.timestamp_key = timestamp_key
        self.reward_key = reward_key
        self.done_key = done_key
        self.success_key = success_key

    def _find_image_key(self, raw: Dict[str, Any]) -> str:
        """Return the first available image key in ``raw``.

        Tries ``self.image_keys`` first, then falls back to
        ``self.image_key``.  Raises ``KeyError`` if none match.
        """
        candidates = self.image_keys or [self.image_key]
        for key in candidates:
            try:
                _get_by_key(raw, key)
                return key
            except KeyError:
                continue
        raise KeyError(
            f"No usable image key found in sample. "
            f"Candidates: {candidates}. "
            f"Available keys: {list(raw.keys())}"
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.base_dataset[index]

        # ── Image ──────────────────────────────────────────────────
        img_key = self._find_image_key(raw)
        img_data = _get_by_key(raw, img_key)
        image = _process_image(img_data, image_size=self.image_size)

        # ── Instruction ─────────────────────────────────────────────
        instr = self.instruction or ""
        token_ids = tokenize(instr, max_len=self.max_text_len)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(
            build_attention_mask(token_ids), dtype=torch.long,
        )

        # ── State / Action ──────────────────────────────────────────
        state = torch.as_tensor(
            _get_by_key(raw, self.state_key), dtype=torch.float32,
        )
        action = torch.as_tensor(
            _get_by_key(raw, self.action_key), dtype=torch.float32,
        )

        sample: Dict[str, Any] = {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "action": action,
            "instruction": instr,
        }

        # ── Optional metadata ───────────────────────────────────────
        meta_keys = [
            (self.episode_index_key, "episode_index"),
            (self.frame_index_key, "frame_index"),
            (self.timestamp_key, "timestamp"),
            (self.reward_key, "next_reward"),
            (self.done_key, "next_done"),
            (self.success_key, "next_success"),
        ]
        for raw_key, sample_key in meta_keys:
            val = _maybe_get_by_key(raw, raw_key)
            if val is not None:
                if isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                elif isinstance(val, np.ndarray):
                    val = val.item()
                sample[sample_key] = val

        return sample


__all__ = ["PushTDatasetAdapter", "_process_image", "_get_by_key"]
