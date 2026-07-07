"""Generic LeRobot dataset loader — works with any LeRobotDataset."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def import_lerobot_dataset():
    """Import ``LeRobotDataset`` across known lerobot version paths.

    Tries, in order:
    1. ``lerobot.datasets.lerobot_dataset.LeRobotDataset``  (v0.4+)
    2. ``lerobot.common.datasets.lerobot_dataset.LeRobotDataset``  (older)

    Returns:
        The ``LeRobotDataset`` class.

    Raises:
        ImportError: If none of the known paths work (detailed error).
    """
    errors: List[str] = []

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except Exception as e:
        errors.append(f"  lerobot.datasets.lerobot_dataset  ->  {type(e).__name__}: {e}")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except Exception as e:
        errors.append(f"  lerobot.common.datasets.lerobot_dataset  ->  {type(e).__name__}: {e}")

    raise ImportError(
        "Could not import LeRobotDataset from any known path.\n"
        "Install or reinstall lerobot:\n"
        "  pip install lerobot\n\n"
        "Tried paths:\n" + "\n".join(errors)
    )


def load_lerobot_samples(
    repo_id: str,
    root: Optional[str] = None,
    max_samples: int = 0,
    validate: bool = False,
) -> List[Dict[str, Any]]:
    """Load samples from any LeRobot dataset.

    This is a **generic** loader — it does not check for any specific
    image key.  Schema inspection is left to the calling code.

    Args:
        repo_id: Hugging Face dataset repo ID (e.g. ``"lerobot/pusht"``).
        root: Optional local root for offline / cached dataset.
        max_samples: If > 0, limit to this many samples.
        validate: If True, verify the returned list is non-empty and each
            element is a dict.

    Returns:
        List of sample dicts.

    Raises:
        ImportError: If ``lerobot`` is not installed.
    """
    LeRobotDataset = import_lerobot_dataset()

    if root:
        ds = LeRobotDataset(repo_id, root=root)
    else:
        ds = LeRobotDataset(repo_id)

    n = min(max_samples, len(ds)) if max_samples > 0 else len(ds)
    samples = [ds[i] for i in range(n)]

    if validate:
        if not samples:
            raise ValueError(f"LeRobot dataset '{repo_id}' returned zero samples.")
        for i, s in enumerate(samples):
            if not isinstance(s, dict):
                raise TypeError(
                    f"Sample {i} is not a dict (got {type(s).__name__}). "
                    "This dataset format may not be compatible."
                )

    return samples


__all__ = [
    "import_lerobot_dataset",
    "load_lerobot_samples",
]
