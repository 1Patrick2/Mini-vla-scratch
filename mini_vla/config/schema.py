"""Minimal config schema checks for Stage 0."""

from __future__ import annotations

from typing import Any

REQUIRED_SECTIONS = ("model", "data", "train")


def validate_config(config: dict[str, Any]) -> None:
    """Validate required top-level config sections."""
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing config section(s): {', '.join(missing)}")

    _require(config["model"], "model", ("name", "state_dim", "action_dim"))
    _require(config["data"], "data", ("dataset_type", "data_root", "batch_size"))
    _require(config["train"], "train", ("epochs", "lr", "device"))


def _require(section: Any, section_name: str, keys: tuple[str, ...]) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"Config section must be a mapping: {section_name}")

    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"Missing {section_name} field(s): {', '.join(missing)}")
