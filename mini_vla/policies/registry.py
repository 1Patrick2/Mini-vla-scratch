"""Policy registry — map policy type strings to classes."""

from __future__ import annotations

from typing import Any, Dict, Type

from mini_vla.policies.base import BasePolicy

_POLICY_REGISTRY: Dict[str, Type[BasePolicy]] = {}


def register_policy(name: str, cls: Type[BasePolicy]) -> None:
    """Register a policy class under *name*.

    Args:
        name: Short identifier (e.g. ``"single_frame_bc"``).
        cls: Subclass of ``BasePolicy``.
    """
    if not issubclass(cls, BasePolicy):
        raise TypeError(
            f"Cannot register {cls.__name__}: must inherit from BasePolicy."
        )
    _POLICY_REGISTRY[name] = cls


def build_policy(config: Dict[str, Any]) -> BasePolicy:
    """Build a policy from configuration.

    The config must contain a ``policy.type`` field (or ``model.policy_type``
    as fallback).  Raises ``ValueError`` for unknown policies.

    Args:
        config: Top-level merged config, or a dict with a ``policy`` subsection.

    Returns:
        A ``BasePolicy`` instance.
    """
    policy_cfg = config.get("policy", config)
    name = policy_cfg.get("type", config.get("model", {}).get("policy_type"))

    if not name:
        raise ValueError(
            "No policy type found in config. "
            "Set policy.type (e.g. 'single_frame_bc') to choose a policy."
        )

    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(_POLICY_REGISTRY))
        raise ValueError(
            f"Unknown policy type: '{name}'. "
            f"Known policies: {known}"
        )

    return cls(config)


def list_policies() -> list[str]:
    """Return sorted list of registered policy names."""
    return sorted(_POLICY_REGISTRY.keys())


__all__ = ["register_policy", "build_policy", "list_policies"]
