"""PushT dataset inspection utilities.

Provides functions to inspect the schema and statistics of a PushT-like
dataset without requiring a full LeRobot installation.  Real dataset
loading is optional — all inspection functions accept plain dict samples.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

# ── Feature key discovery ──────────────────────────────────────────────

DEFAULT_PUSHT_KEYS = {
    "image_key": "observation.image",
    "state_key": "observation.state",
    "action_key": "action",
    "episode_index_key": "episode_index",
    "frame_index_key": "frame_index",
    "timestamp_key": "timestamp",
    "reward_key": "next.reward",
    "done_key": "next.done",
    "success_key": "next.success",
    "task_index_key": "task_index",
}


def discover_feature_keys(
    sample: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return the standard PushT key mapping.

    If a sample is provided, the function verifies that the expected keys
    exist; otherwise it returns the default mapping.
    """
    if sample is not None:
        missing = [k for k in DEFAULT_PUSHT_KEYS.values() if k not in sample]
        if missing:
            raise KeyError(f"Missing expected PushT keys: {missing}")
    return dict(DEFAULT_PUSHT_KEYS)


def summarize_pusht_schema(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the schema of a list of PushT-like samples.

    Returns a dict with:
    - num_samples
    - feature_keys
    - image_shape / dtype
    - state_shape / dtype
    - action_shape / dtype
    - has_{reward,done,success,episode_index,frame_index,timestamp}
    """
    if not samples:
        return {"num_samples": 0}

    s = samples[0]
    schema: Dict[str, Any] = {
        "num_samples": len(samples),
        "feature_keys": list(s.keys()),
    }

    img = s.get(DEFAULT_PUSHT_KEYS["image_key"])
    if img is not None:
        t = torch.as_tensor(img)
        schema["image_shape"] = list(t.shape)
        schema["image_dtype"] = str(t.dtype)

    state = s.get(DEFAULT_PUSHT_KEYS["state_key"])
    if state is not None:
        t = torch.as_tensor(state)
        schema["state_shape"] = list(t.shape)

    action = s.get(DEFAULT_PUSHT_KEYS["action_key"])
    if action is not None:
        t = torch.as_tensor(action)
        schema["action_shape"] = list(t.shape)

    noteys = [
        ("has_reward", "reward_key", DEFAULT_PUSHT_KEYS["reward_key"]),
        ("has_done", "done_key", DEFAULT_PUSHT_KEYS["done_key"]),
        ("has_success", "success_key", DEFAULT_PUSHT_KEYS["success_key"]),
        ("has_episode_index", "episode_index_key", DEFAULT_PUSHT_KEYS["episode_index_key"]),
        ("has_frame_index", "frame_index_key", DEFAULT_PUSHT_KEYS["frame_index_key"]),
        ("has_timestamp", "timestamp_key", DEFAULT_PUSHT_KEYS["timestamp_key"]),
    ]
    for key_name, _, raw_key in noteys:
        schema[key_name] = raw_key in s

    return schema


def compute_state_action_stats(
    samples: Sequence[Dict[str, Any]],
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Compute min/max/mean/std for state and action over the given samples.

    Args:
        samples: List of PushT-like sample dicts.
        max_samples: If > 0, limit to this many samples for speed.

    Returns:
        Dict with ``state`` and ``action`` sub-dicts each containing
        ``min``, ``max``, ``mean``, ``std``.
    """
    pool = samples[:max_samples] if max_samples > 0 else samples
    if not pool:
        return {"state": {}, "action": {}}

    state_key = DEFAULT_PUSHT_KEYS["state_key"]
    action_key = DEFAULT_PUSHT_KEYS["action_key"]

    state_list = [torch.as_tensor(s[state_key]) for s in pool if state_key in s]
    action_list = [torch.as_tensor(s[action_key]) for s in pool if action_key in s]

    def _stats(tensors):
        stacked = torch.stack(tensors)
        return {
            "min": stacked.min(dim=0).values.tolist(),
            "max": stacked.max(dim=0).values.tolist(),
            "mean": stacked.mean(dim=0).tolist(),
            "std": stacked.std(dim=0).tolist(),
        }

    result: Dict[str, Any] = {}
    if state_list:
        result["state"] = _stats(state_list)
    if action_list:
        result["action"] = _stats(action_list)
    return result


def build_pusht_report(
    samples: Sequence[Dict[str, Any]],
    repo_id: str = "lerobot/pusht",
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Build a complete PushT inspection report.

    Args:
        samples: List of PushT-like sample dicts.
        repo_id: Dataset repository name (default ``lerobot/pusht``).
        max_samples: If > 0, limit stats to this many samples.

    Returns:
        A serialisable dict with schema and statistics.
    """
    schema = summarize_pusht_schema(samples)
    stats = compute_state_action_stats(samples, max_samples=max_samples)
    return {
        "repo_id": repo_id,
        **schema,
        "state_stats": stats.get("state", {}),
        "action_stats": stats.get("action", {}),
    }


__all__ = [
    "DEFAULT_PUSHT_KEYS",
    "build_pusht_report",
    "compute_state_action_stats",
    "discover_feature_keys",
    "summarize_pusht_schema",
]
