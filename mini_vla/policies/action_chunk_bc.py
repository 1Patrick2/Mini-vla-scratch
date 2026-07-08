"""ActionChunk BC policy — predict a chunk of future actions (skeleton).

Planned for Stage 8.3.  For now only forwards a placeholder.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from mini_vla.policies.base import BasePolicy
from mini_vla.policies.registry import register_policy


class ActionChunkBCPolicy(BasePolicy):
    """Action Chunking BC policy (skeleton — not yet functional).

    Predicted output shape: ``[B, action_horizon, action_dim]``.
    """

    policy_type = "action_chunk_bc"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("ActionChunkBC not yet implemented in Stage 8.1/8.2")

    def predict_action(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("ActionChunkBC not yet implemented in Stage 8.1/8.2")

    def supports_action_chunk(self) -> bool:
        return True


register_policy("action_chunk_bc", ActionChunkBCPolicy)
