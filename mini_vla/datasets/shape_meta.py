"""shape_meta utilities — resolve state/action dimensions from config.

Read priority:
1. ``shape_meta`` section (``shape_meta.state.dim``, ``shape_meta.action.dim``)
2. ``model`` section (``model.state_dim``, ``model.action_dim``)
3. Raise ``ValueError`` if neither is present — no silent defaults.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _get_from_chain(config: Dict[str, Any], *keys: str) -> Optional[Any]:
    """Walk nested dict; return ``None`` if any key is missing."""
    cur = config
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def get_action_dim(config: Dict[str, Any]) -> int:
    """Resolve the action dimension from config.

    Priority:
    1. ``shape_meta.action.dim``
    2. ``model.action_dim``
    """
    dim = _get_from_chain(config, "shape_meta", "action", "dim")
    if dim is not None:
        return int(dim)
    dim = _get_from_chain(config, "model", "action_dim")
    if dim is not None:
        return int(dim)
    raise ValueError(
        "Could not resolve action_dim from config. "
        "Set shape_meta.action.dim or model.action_dim."
    )


def get_state_dim(config: Dict[str, Any]) -> int:
    """Resolve the state dimension from config.

    Priority:
    1. ``shape_meta.state.dim``
    2. ``model.state_dim``
    """
    dim = _get_from_chain(config, "shape_meta", "state", "dim")
    if dim is not None:
        return int(dim)
    dim = _get_from_chain(config, "model", "state_dim")
    if dim is not None:
        return int(dim)
    raise ValueError(
        "Could not resolve state_dim from config. "
        "Set shape_meta.state.dim or model.state_dim."
    )


def get_obs_horizon(config: Dict[str, Any], default: int = 1) -> int:
    """Resolve the observation horizon."""
    return int(
        _get_from_chain(config, "shape_meta", "obs_horizon")
        or config.get("model", {}).get("obs_horizon")
        or default
    )


def get_action_horizon(config: Dict[str, Any], default: int = 1) -> int:
    """Resolve the action horizon (number of future steps to predict)."""
    return int(
        _get_from_chain(config, "shape_meta", "action_horizon")
        or config.get("model", {}).get("action_horizon")
        or default
    )


__all__ = [
    "get_action_dim",
    "get_state_dim",
    "get_obs_horizon",
    "get_action_horizon",
]
