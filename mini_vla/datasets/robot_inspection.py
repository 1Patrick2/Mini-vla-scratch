"""Generic robot dataset inspection utilities.

Works with any LeRobot-like dataset via DatasetSpec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from mini_vla.datasets.registry import get_dataset_spec
from mini_vla.datasets.spec import DatasetSpec


def _find_first_match(
    available: Sequence[str],
    candidates: Sequence[str],
) -> Optional[str]:
    """Return the first candidate key that exists in ``available``."""
    for key in candidates:
        if key in available:
            return key
    return None


def inspect_robot_dataset(
    samples: Sequence[Dict[str, Any]],
    spec: DatasetSpec,
    repo_id: str = "",
    loader: str = "lerobot",
) -> Dict[str, Any]:
    """Build an inspection report for a robot dataset.

    Args:
        samples: List of sample dicts (at least one).
        spec: The ``DatasetSpec`` for this dataset.
        repo_id: Override repo ID (falls back to ``spec.repo_id``).
        loader: Loader backend name.

    Returns:
        A serialisable report dict containing schema and key mapping info.
    """
    report: Dict[str, Any] = {
        "dataset_name": spec.name,
        "repo_id": repo_id or spec.repo_id,
        "loader": loader,
        "num_samples": len(samples),
        "supports_training": spec.supports_training,
        "supports_evaluation": spec.supports_evaluation,
    }

    if not samples:
        return report

    s0 = samples[0]
    available = list(s0.keys())
    report["available_keys"] = available

    # Match keys
    matched_image = _find_first_match(available, spec.image_keys)
    matched_state = _find_first_match(available, spec.state_keys)
    matched_action = _find_first_match(available, spec.action_keys)
    matched_language = _find_first_match(available, spec.language_keys)

    report["matched_image_key"] = matched_image
    report["matched_state_key"] = matched_state
    report["matched_action_key"] = matched_action
    report["matched_language_key"] = matched_language

    # Shapes
    if matched_image:
        t = torch.as_tensor(s0[matched_image])
        report["image_shape"] = list(t.shape)
    if matched_state:
        t = torch.as_tensor(s0[matched_state])
        report["state_shape"] = list(t.shape)
    if matched_action:
        t = torch.as_tensor(s0[matched_action])
        report["action_shape"] = list(t.shape)

    # Instruction example
    if matched_language:
        instr = s0.get(matched_language)
        if instr is not None:
            report["instruction_example"] = str(instr)
    if "instruction_example" not in report and spec.default_instruction:
        report["instruction_example"] = spec.default_instruction

    # Recommendation
    rec_parts: List[str] = []
    if matched_image:
        rec_parts.append("vision OK")
    else:
        rec_parts.append("no matched image key")
    if matched_state:
        rec_parts.append(f"state {report.get('state_shape')}")
    else:
        rec_parts.append("no matched state key")
    if matched_action:
        rec_parts.append(f"action {report.get('action_shape')}")
    else:
        rec_parts.append("no matched action key")
    report["matched_summary"] = ", ".join(rec_parts)

    return report


def _make_mock_samples(
    spec: DatasetSpec,
    n: int = 4,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate mock samples that follow a spec's key candidates.

    Only the first candidate for each key group is populated.
    """
    rng = np.random.RandomState(seed)
    samples = []
    for i in range(n):
        sample: Dict[str, Any] = {}
        if spec.image_keys:
            sample[spec.image_keys[0]] = rng.randint(
                0, 256, (3, 96, 96), dtype=np.uint8,
            )
        if spec.state_keys:
            sample[spec.state_keys[0]] = rng.randn(2).astype(np.float32)
        if spec.action_keys:
            sample[spec.action_keys[0]] = rng.randn(2).astype(np.float32)
        if spec.language_keys:
            sample[spec.language_keys[0]] = f"mock instruction {i}"
        sample["episode_index"] = i // 2
        sample["frame_index"] = i % 2
        samples.append(sample)
    return samples


def inspect_robot_dataset_from_registry(
    dataset_name: str,
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    repo_id: str = "",
    loader: str = "lerobot",
) -> Dict[str, Any]:
    """Helper: resolve spec from registry and inspect.

    If ``samples`` is None, generates mock samples.
    """
    spec = get_dataset_spec(dataset_name)
    if samples is None:
        samples = _make_mock_samples(spec)
    return inspect_robot_dataset(samples, spec, repo_id=repo_id, loader=loader)


__all__ = [
    "inspect_robot_dataset",
    "inspect_robot_dataset_from_registry",
]
