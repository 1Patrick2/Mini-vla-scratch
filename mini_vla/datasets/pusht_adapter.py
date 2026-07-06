"""PushT dataset adapter — converts PushT samples to MiniVLA format.

Usage:
    adapter = PushTDatasetAdapter(raw_dataset)
    sample = adapter[0]  # MiniVLA-format sample dict
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from mini_vla.datasets.pusht_inspection import DEFAULT_PUSHT_KEYS
from mini_vla.datasets.transforms import build_attention_mask, tokenize


class PushTDatasetAdapter:
    """Adapt PushT-like samples to MiniVLA format.

    Args:
        base_dataset: A list, list-like, or sequence of PushT sample dicts.
        image_key: Key for the image in the PushT sample.
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

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.base_dataset[index]

        # ── Image ──────────────────────────────────────────────────
        img_data = raw[self.image_key]
        image = self._process_image(img_data)

        # ── Instruction ─────────────────────────────────────────────
        instr = self.instruction or ""
        token_ids = tokenize(instr, max_len=self.max_text_len)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(
            build_attention_mask(token_ids), dtype=torch.long,
        )

        # ── State / Action ──────────────────────────────────────────
        state = torch.as_tensor(raw[self.state_key], dtype=torch.float32)
        action = torch.as_tensor(raw[self.action_key], dtype=torch.float32)

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
            if raw_key in raw:
                val = raw[raw_key]
                if isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                elif isinstance(val, np.ndarray):
                    val = val.item()
                sample[sample_key] = val

        return sample

    def _process_image(self, img_data: Any) -> torch.Tensor:
        """Convert a PushT image to a CHW float32 tensor in [0, 1].

        Supports:
        - HWC uint8 numpy array (96, 96, 3)
        - HWC float32 numpy array (96, 96, 3)
        - CHW torch.Tensor (3, H, W)
        - PIL Image
        """
        if isinstance(img_data, Image.Image):
            pil = img_data
        elif isinstance(img_data, torch.Tensor):
            if img_data.ndim == 3 and img_data.shape[0] in (1, 3):
                # Already CHW — resize if needed
                if img_data.shape[1] != self.image_size:
                    pil = Image.fromarray(
                        img_data.permute(1, 2, 0).byte().numpy()
                    )
                    pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)
                    return torch.as_tensor(
                        np.array(pil), dtype=torch.float32
                    ).permute(2, 0, 1).div(255.0).clamp(0, 1)
                # Already correct size — normalise to [0, 1]
                result = img_data.float()
                if result.is_floating_point():
                    result = result.clamp(0, 1).div(255.0)
                return result.div(255.0).clamp(0, 1)
            # HWC tensor
            pil = Image.fromarray(img_data.byte().numpy())
        elif isinstance(img_data, np.ndarray):
            if img_data.ndim == 3 and img_data.shape[-1] in (1, 3):
                # HWC
                pil = Image.fromarray(img_data)
            elif img_data.ndim == 3 and img_data.shape[0] in (1, 3):
                # CHW
                pil = Image.fromarray(img_data.transpose(1, 2, 0))
            else:
                raise ValueError(f"Unexpected image array shape: {img_data.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(img_data)}")

        # Resize to target size
        if pil.size != (self.image_size, self.image_size):
            pil = pil.resize((self.image_size, self.image_size), Image.BILINEAR)

        arr = np.array(pil, dtype=np.float32) / 255.0
        # (H, W, C) → (C, H, W)
        return torch.from_numpy(arr).permute(2, 0, 1).clamp(0, 1)


__all__ = ["PushTDatasetAdapter"]
