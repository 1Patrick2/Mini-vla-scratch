"""Dataset specification — declares a dataset's schema for the Zoo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative specification of a robot dataset's key schema.

    Attributes:
        name: Short identifier (e.g. ``"pusht"``).
        repo_id: Hugging Face repo ID (e.g. ``"lerobot/pusht"``).
        loader: Default loader backend (``"lerobot"`` or ``"hf_datasets"``).
        image_keys: Candidate keys for the visual observation, tried in order.
        state_keys: Candidate keys for the robot state.
        action_keys: Candidate keys for the action.
        language_keys: Candidate keys for the language instruction.
        episode_index_key: Key for episode index metadata.
        frame_index_key: Key for frame index metadata.
        default_instruction: Fixed instruction used when no language key is found.
        supports_training: Whether full BC training is supported in Stage 6.
        supports_evaluation: Whether offline evaluation is supported in Stage 6.
        notes: Free-form notes about dataset status or limitations.
    """

    name: str
    repo_id: str
    loader: str = "lerobot"
    image_keys: tuple[str, ...] = ()
    state_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = ()
    language_keys: tuple[str, ...] = ()
    episode_index_key: str = "episode_index"
    frame_index_key: str = "frame_index"
    default_instruction: Optional[str] = None
    supports_training: bool = False
    supports_evaluation: bool = False
    notes: str = ""


__all__ = ["DatasetSpec"]
