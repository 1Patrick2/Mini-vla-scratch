"""Utilities for accessing flat and nested dotted keys in sample dicts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def get_by_key(sample: Dict[str, Any], key: str) -> Any:
    """Get a value by dotted key, supporting flat and nested access.

    Tries ``sample["observation.image"]`` first, then
    ``sample["observation"]["image"]``, then deeper nesting.

    Raises:
        KeyError: If none of the access paths find the key.
    """
    if key in sample:
        return sample[key]
    parts = key.split(".")
    current: Any = sample
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(
                f"Key '{key}' not found in sample (tried flat and nested). "
                f"Available keys: {list(sample.keys())}"
            )
    return current


def has_key(sample: Dict[str, Any], key: str) -> bool:
    """Check if a dotted key exists in a sample (flat or nested)."""
    try:
        get_by_key(sample, key)
        return True
    except KeyError:
        return False


def find_first_key(sample: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate key that exists in the sample.

    Returns ``None`` if none match.
    """
    for key in candidates:
        if has_key(sample, key):
            return key
    return None


def list_available_keys(
    sample: Dict[str, Any],
    prefix: str = "",
    flatten_nested: bool = True,
) -> List[str]:
    """List all keys in a sample, optionally flattening nested dicts.

    A nested dict like ``{"observation": {"image": ..., "state": ...}}``
    produces ``["observation.image", "observation.state"]`` when
    ``flatten_nested=True``.
    """
    keys: List[str] = []
    for k, v in sample.items():
        full = f"{prefix}.{k}" if prefix else k
        if flatten_nested and isinstance(v, dict):
            keys.extend(list_available_keys(v, prefix=full))
        else:
            keys.append(full)
    return keys
