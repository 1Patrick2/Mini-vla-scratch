"""Adapt model actions into safe robot-facing action values."""

from __future__ import annotations


def clip_action(action: list[float], limit: float) -> list[float]:
    """Clip every action dimension to [-limit, limit]."""
    if limit < 0:
        raise ValueError("limit must be non-negative")

    return [max(-limit, min(limit, value)) for value in action]
