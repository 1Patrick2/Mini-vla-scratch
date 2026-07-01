"""MiniVLA model builder — construct a model from a configuration dict.

Usage:
    from mini_vla.models import build_model
    model = build_model(config_dict)
    action_pred = model(batch)

Strict type and dimension checking is performed.  Invalid config values
raise ``ValueError`` with a clear message.
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

    Raises:
        ValueError: If an unsupported component type or dimension mismatch
            is detected.
    """
    cfg = config.get("model", config)

    # ── Model name validation ────────────────────────────────────
    model_name = cfg.get("name", "mini_vla")
    if model_name != "mini_vla":
        raise ValueError(
            f"Unsupported model.name: '{model_name}'. Expected 'mini_vla'."
        )

    action_dim = cfg.get("action_dim", 2)

    # ── Vision encoder ──────────────────────────────────────────
    ve_cfg = cfg.get("vision_encoder", {})
    if ve_cfg.get("type", "small_cnn") != "small_cnn":
        raise ValueError(
            f"Unsupported vision_encoder.type: '{ve_cfg.get('type')}'. "
            f"Expected 'small_cnn'."
        )
    ve_output = ve_cfg.get("output_dim", 128)
    vision_encoder = SmallCNNVisionEncoder(output_dim=ve_output)

    # ── Text encoder ────────────────────────────────────────────
    te_cfg = cfg.get("text_encoder", {})
    te_type = te_cfg.get("type", "mock_llm")
    te_output = te_cfg.get("output_dim", 128)

    if te_type == "mock_llm":
        text_encoder = MockLLMTextEncoder(
            vocab_size=te_cfg.get("vocab_size", 128),
            hidden_dim=te_cfg.get("hidden_dim", 128),
            output_dim=te_output,
        )
    elif te_type == "hf_llm":
        raise ValueError(
            "LLMTextEncoder (type='hf_llm') is reserved for future use. "
            "Please install transformers and use it directly."
        )
    else:
        raise ValueError(
            f"Unsupported text_encoder.type: '{te_type}'. Expected 'mock_llm'."
        )

    if te_cfg.get("freeze", False):
        for param in text_encoder.parameters():
            param.requires_grad = False

    # ── State encoder ───────────────────────────────────────────
    se_cfg = cfg.get("state_encoder", {})
    se_output = se_cfg.get("output_dim", 128)
    state_encoder = StateEncoder(
        input_dim=se_cfg.get("input_dim", 2),
        hidden_dim=se_cfg.get("hidden_dim", 64),
        output_dim=se_output,
    )

    # ── Dimensionality cross-check ──────────────────────────────
    expected_fusion_input = ve_output + te_output + se_output
    fu_cfg = cfg.get("fusion", {})
    fu_input = fu_cfg.get("input_dim", 384)

    if fu_input != expected_fusion_input:
        raise ValueError(
            f"fusion.input_dim ({fu_input}) does not match "
            f"vision_encoder.output_dim ({ve_output}) + "
            f"text_encoder.output_dim ({te_output}) + "
            f"state_encoder.output_dim ({se_output}) = {expected_fusion_input}."
        )

    # ── Fusion ──────────────────────────────────────────────────
    if fu_cfg.get("type", "concat_mlp") != "concat_mlp":
        raise ValueError(
            f"Unsupported fusion.type: '{fu_cfg.get('type')}'. "
            f"Expected 'concat_mlp'."
        )
    fu_output = fu_cfg.get("output_dim", 128)
    fusion = FusionMLP(
        input_dim=fu_input,
        hidden_dim=fu_cfg.get("hidden_dim", 256),
        output_dim=fu_output,
    )

    # ── Action head ─────────────────────────────────────────────
    ah_cfg = cfg.get("action_head", {})
    ah_input = ah_cfg.get("input_dim", 128)

    if ah_input != fu_output:
        raise ValueError(
            f"action_head.input_dim ({ah_input}) does not match "
            f"fusion.output_dim ({fu_output})."
        )

    action_head = ActionHead(
        input_dim=ah_input,
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
