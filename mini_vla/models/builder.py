"""MiniVLA model builder — construct a model from a configuration dict.

Usage:
    from mini_vla.models.builder import build_model
    model = build_model(config_dict)
    action_pred = model(batch)
"""

from __future__ import annotations

from typing import Any, Dict

from mini_vla.models.action_head import ActionHead
from mini_vla.models.fusion import FusionMLP
from mini_vla.models.language_encoder import MockLLMTextEncoder
from mini_vla.models.mini_vla import MiniVLA
from mini_vla.models.state_encoder import StateEncoder
from mini_vla.models.vision_encoder import SmallCNNVisionEncoder


def build_model(config: Dict[str, Any]) -> MiniVLA:
    """Build a MiniVLA model from a configuration dict.

    The expected structure matches ``configs/model/mini_vla_cnn_llm.yaml``.

    Args:
        config: Model configuration with sections for each component.

    Returns:
        A fully assembled ``MiniVLA`` model.
    """
    cfg = config.get("model", config)
    action_dim = cfg.get("action_dim", 2)

    # Vision encoder
    ve_cfg = cfg.get("vision_encoder", {})
    vision_encoder = SmallCNNVisionEncoder(
        output_dim=ve_cfg.get("output_dim", 128),
    )

    # Text encoder (currently only mock_llm, extendable)
    te_cfg = cfg.get("text_encoder", {})
    text_encoder = MockLLMTextEncoder(
        vocab_size=te_cfg.get("vocab_size", 128),
        hidden_dim=te_cfg.get("hidden_dim", 128),
        output_dim=te_cfg.get("output_dim", 128),
    )

    # State encoder
    se_cfg = cfg.get("state_encoder", {})
    state_encoder = StateEncoder(
        input_dim=se_cfg.get("input_dim", 2),
        hidden_dim=se_cfg.get("hidden_dim", 64),
        output_dim=se_cfg.get("output_dim", 128),
    )

    # Fusion
    fu_cfg = cfg.get("fusion", {})
    fusion = FusionMLP(
        input_dim=fu_cfg.get("input_dim", 384),
        hidden_dim=fu_cfg.get("hidden_dim", 256),
        output_dim=fu_cfg.get("output_dim", 128),
    )

    # Action head
    ah_cfg = cfg.get("action_head", {})
    action_head = ActionHead(
        input_dim=ah_cfg.get("input_dim", 128),
        hidden_dim=ah_cfg.get("hidden_dim", 128),
        action_dim=action_dim,
    )

    return MiniVLA(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        state_encoder=state_encoder,
        fusion=fusion,
        action_head=action_head,
    )


__all__ = ["build_model"]
