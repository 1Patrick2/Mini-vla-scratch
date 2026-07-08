"""Dataset transforms — history, delta, etc.

Each transform wraps an existing dataset and adds new fields.
Transforms are composable and can be chained.

Tokenization utilities have been moved to ``tokenization.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from mini_vla.datasets.tokenization import (  # noqa: F401
    ID_TO_TOKEN,
    VOCAB,
    build_attention_mask,
    decode,
    tokenize,
)


class HistoryDatasetWrapper:
    """Wraps a dataset and adds previous-frame state/action fields.

    Input samples must have ``episode_index`` and ``frame_index`` for
    correct temporal ordering.

    Args:
        dataset: Sequence of sample dicts.
        history_size: Number of previous frames (1 only in Stage 7).
        include_prev_state: Include previous state fields.
        include_prev_action: Include previous action fields.
        first_frame_prev_state: ``"current"`` (copy current) or ``"zero"``.
        first_frame_prev_action: ``"zero"``.
        concat_to_state: If True, set ``sample["state"]`` = concat of
            ``[state, prev_state, prev_action]``.
        state_dim: Original state dimension before concat.
    """

    def __init__(
        self,
        dataset: Sequence[Dict[str, Any]],
        history_size: int = 1,
        include_prev_state: bool = True,
        include_prev_action: bool = True,
        first_frame_prev_state: str = "current",
        first_frame_prev_action: str = "zero",
        concat_to_state: bool = False,
        state_dim: int = 2,
    ):
        if history_size != 1:
            raise ValueError(f"history_size={history_size} not supported (only 1)")
        self.dataset = dataset
        self.include_prev_state = include_prev_state
        self.include_prev_action = include_prev_action
        self.first_frame_prev_state = first_frame_prev_state
        self.first_frame_prev_action = first_frame_prev_action
        self.concat_to_state = concat_to_state
        self.state_dim = state_dim
        self._index = self._build_index()

    def _build_index(self) -> list[int]:
        indexed = []
        for i in range(len(self.dataset)):
            s = self.dataset[i]
            ep = s.get("episode_index", -1)
            fr = s.get("frame_index", i)
            if isinstance(ep, torch.Tensor):
                ep = ep.item()
            if isinstance(fr, torch.Tensor):
                fr = fr.item()
            indexed.append((int(ep) if ep is not None else -1,
                            int(fr) if fr is not None else i, i))
        indexed.sort(key=lambda x: (x[0], x[1]))
        return [idx for (_, _, idx) in indexed]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base = self.dataset[self._index[index]]
        # Deep-copy tensors to avoid in-place pollution of underlying dataset
        sample: Dict[str, Any] = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in base.items()
        }
        current_ep = sample.get("episode_index")
        is_first = True
        prev_pos = None

        if current_ep is not None and index > 0:
            prev_idx = self._index[index - 1]
            prev_s = self.dataset[prev_idx]
            prev_ep = prev_s.get("episode_index")
            if isinstance(prev_ep, torch.Tensor):
                prev_ep = prev_ep.item()
            if prev_ep == current_ep:
                prev_pos = prev_idx
                is_first = False

        if prev_pos is not None:
            ps = self.dataset[prev_pos]
            p_state_r = ps.get("state_raw", ps["state"]).clone()
            p_act_r = ps.get("action_raw", ps["action"]).clone()
            p_state_n = ps.get("state_normalized", ps["state"]).clone()
            p_act_n = ps.get("action_normalized", ps["action"]).clone()
        else:
            p_state_r = sample.get("state_raw", sample["state"]).clone()
            p_act_r = torch.zeros_like(sample.get("action_raw", sample["action"]))
            p_state_n = sample.get("state_normalized", sample["state"]).clone()
            p_act_n = torch.zeros_like(sample.get("action_normalized", sample["action"]))

        sample["is_first_frame"] = is_first
        if self.include_prev_state:
            sample["prev_state_raw"] = p_state_r
            sample["prev_state_normalized"] = p_state_n
            sample["prev_state"] = p_state_n if "state_normalized" in sample else p_state_r
        if self.include_prev_action:
            sample["prev_action_raw"] = p_act_r
            sample["prev_action_normalized"] = p_act_n
            sample["prev_action"] = p_act_n if "action_normalized" in sample else p_act_r

        if self.concat_to_state:
            cur_st = sample.get("state_normalized", sample["state"])
            prv_st = sample.get("prev_state_normalized", sample["prev_state"])
            prv_ac = sample.get("prev_action_normalized", sample["prev_action"])
            sample["state"] = torch.cat([cur_st, prv_st, prv_ac])

        return sample


class DeltaActionTargetWrapper:
    """Transform training target from absolute action to delta action.

    Computes ``delta_action = action - prev_action`` and replaces
    ``sample["action"]`` with this delta, while preserving the original
    action in ``target_action_*`` fields for evaluation.

    Requires sample to have ``action``, ``prev_action``,
    and optionally ``action_raw`` / ``action_normalized`` /
    ``prev_action_raw`` / ``prev_action_normalized``.
    """

    def __init__(self, dataset: Sequence[Dict[str, Any]]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]

        # Preserve original targets
        sample["target_type"] = "delta_action"
        sample["target_action_raw"] = sample.get("action_raw", sample["action"]).clone()
        if "action_normalized" in sample:
            sample["target_action_normalized"] = sample["action_normalized"].clone()
        else:
            sample["target_action_normalized"] = sample["action"].clone()

        # Compute delta in available spaces
        if "action_normalized" in sample and "prev_action_normalized" in sample:
            sample["delta_action_normalized"] = (
                sample["action_normalized"] - sample["prev_action_normalized"]
            )
        if "action_raw" in sample and "prev_action_raw" in sample:
            sample["delta_action_raw"] = (
                sample["action_raw"] - sample["prev_action_raw"]
            )

        # Set training target: prefer normalized delta, fall back to raw
        if "delta_action_normalized" in sample:
            sample["action"] = sample["delta_action_normalized"]
        elif "delta_action_raw" in sample:
            sample["action"] = sample["delta_action_raw"]

        return sample
