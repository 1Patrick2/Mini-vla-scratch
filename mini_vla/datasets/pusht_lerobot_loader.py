"""Load PushT samples with real image frames via LeRobotDataset.

This is the **real vision** path for PushT.  ``datasets.load_dataset``
(Hugging Face ``datasets``) does *not* return ``observation.image`` for
``lerobot/pusht`` — it only provides state/action/metadata.  Use this
module instead to get actual video frames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_IMAGE_KEY_CANDIDATES = [
    "observation.image",
    "observation.images.top",
    "observation.images.main",
    "image",
]


def load_pusht_lerobot(
    repo_id: str = "lerobot/pusht",
    root: Optional[str] = None,
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    """Load real PushT samples with images via LeRobotDataset.

    Args:
        repo_id: Hugging Face dataset repo ID (default ``lerobot/pusht``).
        root: Optional local root for offline / cached dataset.
        max_samples: If > 0, limit to this many samples.

    Returns:
        List of sample dicts.  Each sample contains ``observation.image``
        (numpy uint8 array, shape ``[96, 96, 3]``), ``observation.state``,
        and ``action``, plus metadata keys.

    Raises:
        ImportError: If ``lerobot`` is not installed.
        KeyError: If no usable image key is found in the first sample.
    """
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        raise ImportError(
            "Real PushT vision loading requires 'lerobot'. "
            "Install it with: pip install lerobot"
        ) from None

    if root:
        ds = LeRobotDataset(repo_id, root=root)
    else:
        ds = LeRobotDataset(repo_id)

    n = min(max_samples, len(ds)) if max_samples > 0 else len(ds)
    samples = [ds[i] for i in range(n)]

    # Verify the first sample has an image
    if samples:
        _check_image_key(samples[0])

    return samples


def _check_image_key(sample: Dict[str, Any]) -> str:
    """Verify a sample contains at least one usable image key.

    Returns the matched key name.

    Raises:
        KeyError: If none of the candidate keys hold an image.
    """
    for key in _IMAGE_KEY_CANDIDATES:
        if key in sample:
            return key
    raise KeyError(
        "LeRobotDataset sample does not contain a usable image key "
        f"({_IMAGE_KEY_CANDIDATES}). "
        f"Available keys: {list(sample.keys())}\n"
        "This is not a valid vision PushT sample for MiniVLA. "
    )


def find_image_key(sample: Dict[str, Any]) -> Optional[str]:
    """Return the first matching image key, or ``None``."""
    for key in _IMAGE_KEY_CANDIDATES:
        if key in sample:
            return key
    return None


__all__ = [
    "find_image_key",
    "load_pusht_lerobot",
]
