"""Dataset registry — maps dataset names to DatasetSpec objects."""

from __future__ import annotations

from mini_vla.datasets.spec import DatasetSpec

# ── Registry ──────────────────────────────────────────────────────────

_REGISTRY: dict[str, DatasetSpec] = {}


def _register(spec: DatasetSpec) -> None:
    _REGISTRY[spec.name] = spec


# ── PushT ─────────────────────────────────────────────────────────────

_register(DatasetSpec(
    name="pusht",
    repo_id="lerobot/pusht",
    image_keys=(
        "observation.image",
        "observation.images.top",
        "observation.images.main",
        "image",
    ),
    state_keys=(
        "observation.state",
        "state",
    ),
    action_keys=("action",),
    language_keys=(
        "task",
        "language_instruction",
        "instruction",
    ),
    default_instruction="push the T block to the target",
    supports_training=True,
    supports_evaluation=True,
    notes="Full train/eval support. Vision (96×96), state [2], action [2].",
))

# ── ALOHA sim transfer cube ──────────────────────────────────────────

_register(DatasetSpec(
    name="aloha_sim_transfer_cube",
    repo_id="lerobot/aloha_sim_transfer_cube_scripted",
    image_keys=(
        "observation.images.top",
        "observation.images.cam_high",
        "observation.image",
        "image",
    ),
    state_keys=(
        "observation.state",
        "observation.qpos",
        "state",
    ),
    action_keys=("action",),
    language_keys=(
        "task",
        "language_instruction",
        "instruction",
    ),
    default_instruction="transfer the cube",
    supports_training=False,
    supports_evaluation=False,
    notes="Stage 6 inspect/adapter smoke only. Action dim may differ.",
))

# ── LIBERO ────────────────────────────────────────────────────────────

_register(DatasetSpec(
    name="libero",
    repo_id="lerobot/libero",
    image_keys=(
        "observation.images.agentview",
        "observation.images.eye_in_hand",
        "observation.image",
    ),
    state_keys=(
        "observation.state",
        "state",
    ),
    action_keys=("action",),
    language_keys=(
        "task",
        "language_instruction",
        "instruction",
    ),
    default_instruction=None,
    supports_training=False,
    supports_evaluation=False,
    notes="Stage 6 feasibility inspect only. Multi-task, higher complexity.",
))

# ── Public API ────────────────────────────────────────────────────────


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the ``DatasetSpec`` for a given dataset name.

    Args:
        name: Dataset name (e.g. ``"pusht"``).

    Returns:
        The matching ``DatasetSpec``.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available: {available}"
        )
    return _REGISTRY[name]


def list_dataset_names() -> list[str]:
    """Return all registered dataset names."""
    return sorted(_REGISTRY.keys())


__all__ = [
    "get_dataset_spec",
    "list_dataset_names",
]
