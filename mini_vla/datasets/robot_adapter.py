"""Generic robot dataset adapter — converts any LeRobotDataset to MiniVLA format."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from mini_vla.datasets.key_utils import find_first_key, get_by_key
from mini_vla.datasets.normalization import ActionNormalizer
from mini_vla.datasets.spec import DatasetSpec
from mini_vla.datasets.transforms import build_attention_mask, tokenize


def _process_image(img_data: Any, image_size: int = 64) -> torch.Tensor:
    """Convert image data to a CHW float32 tensor in [0, 1].

    Supports HWC numpy (uint8/float), CHW torch, and PIL.
    """
    if isinstance(img_data, Image.Image):
        pil = img_data

    elif isinstance(img_data, torch.Tensor):
        if img_data.ndim == 3 and img_data.shape[0] in (1, 3):
            # CHW
            img = img_data.float()
            if img.max() > 1.0:
                img = img / 255.0
            img = img.clamp(0, 1)
            if img.shape[1] != image_size or img.shape[2] != image_size:
                pil = Image.fromarray((img.permute(1, 2, 0) * 255).byte().numpy())
                pil = pil.resize((image_size, image_size), Image.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                return torch.from_numpy(arr).permute(2, 0, 1).clamp(0, 1)
            return img
        # HWC tensor
        pil = Image.fromarray(img_data.byte().numpy())

    elif isinstance(img_data, np.ndarray):
        if img_data.ndim == 3 and img_data.shape[-1] in (1, 3):
            arr = img_data.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = arr.clip(0, 1)
            pil = Image.fromarray((arr * 255).astype(np.uint8))
        elif img_data.ndim == 3 and img_data.shape[0] in (1, 3):
            arr = img_data.transpose(1, 2, 0).astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = arr.clip(0, 1)
            pil = Image.fromarray((arr * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unexpected image shape: {img_data.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(img_data)}")

    if pil.size != (image_size, image_size):
        pil = pil.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).clamp(0, 1)


def _read_instruction(sample: Dict[str, Any], spec: DatasetSpec) -> str:
    """Read instruction from a sample, falling back to ``default_instruction``."""
    from mini_vla.datasets.key_utils import find_first_key

    key = find_first_key(sample, spec.language_keys)
    if key is not None:
        try:
            val = get_by_key(sample, key)
            if val is not None and str(val).strip():
                return str(val)
        except KeyError:
            pass
    return spec.default_instruction or ""


class BaseRobotDatasetAdapter:
    """Generic robot dataset adapter — maps any LeRobotDataset to MiniVLA format.

    Args:
        base_dataset: A list, list-like, or sequence of sample dicts.
        spec: ``DatasetSpec`` describing the dataset schema.
        image_size: Target image size (default 64).
        max_text_len: Max token length for instruction (default 16).
        normalizer: Optional ``ActionNormalizer`` for state/action.
        strict: If True (default), missing required keys raise errors.
            ``strict=False`` is for inspect/smoke/debug only, **not** for training.
        allow_missing_action: If True and ``strict=False``, missing action
            is replaced with a zero tensor instead of raising.  Default False.
            Never set this to True for training — action is required for BC.
    """

    def __init__(
        self,
        base_dataset: Sequence[Dict[str, Any]],
        spec: DatasetSpec,
        image_size: int = 64,
        max_text_len: int = 16,
        normalizer: Optional[ActionNormalizer] = None,
        strict: bool = True,
        allow_missing_action: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.spec = spec
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.normalizer = normalizer
        self.strict = strict
        self.allow_missing_action = allow_missing_action

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.base_dataset[index]

        # ── Image ──────────────────────────────────────────────────
        img_key = find_first_key(raw, self.spec.image_keys)
        if img_key is None:
            if self.strict:
                raise KeyError(
                    f"No image key found among {self.spec.image_keys}. "
                    f"Available keys: {list(raw.keys())}"
                )
            # Create a blank image for non-strict (inspect) mode
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        else:
            img_data = get_by_key(raw, img_key)
            image = _process_image(img_data, image_size=self.image_size)

        # ── Instruction ─────────────────────────────────────────────
        instr = _read_instruction(raw, self.spec)
        token_ids = tokenize(instr, max_len=self.max_text_len)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(
            build_attention_mask(token_ids), dtype=torch.long,
        )

        # ── State ──────────────────────────────────────────────────
        state_key = find_first_key(raw, self.spec.state_keys)
        if state_key is None:
            if self.strict:
                raise KeyError(
                    f"No state key found among {self.spec.state_keys}. "
                    f"Available keys: {list(raw.keys())}"
                )
            state_raw = torch.zeros(2, dtype=torch.float32)
        else:
            state_raw = torch.as_tensor(
                get_by_key(raw, state_key), dtype=torch.float32,
            )

        # ── Action ─────────────────────────────────────────────────
        action_key = find_first_key(raw, self.spec.action_keys)
        if action_key is None:
            if self.strict or not self.allow_missing_action:
                raise KeyError(
                    f"No action key found among {self.spec.action_keys}. "
                    f"Available keys: {list(raw.keys())}. "
                    "Action is required for BC training."
                )
            action_raw = torch.zeros(2, dtype=torch.float32)
        else:
            action_raw = torch.as_tensor(
                get_by_key(raw, action_key), dtype=torch.float32,
            )

        # ── Normalize ──────────────────────────────────────────────
        # Convention:
        #   sample["state"] / sample["action"] = the training space
        #   sample["state_raw"] / sample["action_raw"] = always raw
        #   If normalizer exists:
        #     state/action = normalized
        #     state_normalized/action_normalized also available
        sample: Dict[str, Any] = {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state_raw": state_raw,
            "action_raw": action_raw,
        }

        if self.normalizer:
            state_norm = self.normalizer.normalize_state(state_raw)
            action_norm = self.normalizer.normalize_action(action_raw)
            sample["state"] = state_norm
            sample["action"] = action_norm
            sample["state_normalized"] = state_norm
            sample["action_normalized"] = action_norm
        else:
            sample["state"] = state_raw
            sample["action"] = action_raw

        # ── Instruction ─────────────────────────────────────────────
        sample["instruction"] = instr

        # ── Metadata ───────────────────────────────────────────────
        ep_key = self.spec.episode_index_key
        fr_key = self.spec.frame_index_key
        if ep_key in raw:
            sample["episode_index"] = raw[ep_key]
        if fr_key in raw:
            sample["frame_index"] = raw[fr_key]

        return sample


__all__ = ["BaseRobotDatasetAdapter"]
