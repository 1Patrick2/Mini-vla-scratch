"""Policy abstraction layer for MiniVLA.

Each policy wraps a MiniVLA model and provides a uniform interface for
training (``compute_loss``) and inference (``predict_action``).
"""

from mini_vla.policies.action_chunk_bc import ActionChunkBCPolicy
from mini_vla.policies.base import BasePolicy
from mini_vla.policies.delta_action_bc import DeltaActionBCPolicy
from mini_vla.policies.history_bc import HistoryBCPolicy
from mini_vla.policies.registry import build_policy, list_policies, register_policy
from mini_vla.policies.single_frame_bc import SingleFrameBCPolicy

__all__ = [
    "ActionChunkBCPolicy",
    "BasePolicy",
    "SingleFrameBCPolicy",
    "HistoryBCPolicy",
    "DeltaActionBCPolicy",
    "register_policy",
    "build_policy",
    "list_policies",
]
