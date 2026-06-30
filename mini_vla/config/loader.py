"""Load and merge simple YAML configuration files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mini_vla.config.schema import validate_config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a simple YAML config and merge its optional base config."""
    config_path = Path(path)
    data = _read_yaml(config_path)
    base_ref = data.pop("base", None)

    if base_ref is None:
        merged = data
    else:
        base_path = (config_path.parent / str(base_ref)).resolve()
        merged = _deep_merge(load_config(base_path), data)

    validate_config(merged)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return _parse_simple_yaml(path.read_text(encoding="utf-8"), path)


def _parse_simple_yaml(text: str, path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid config line {path}:{line_number}: {raw_line}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation at {path}:{line_number}")

        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value.strip('"\'')


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result