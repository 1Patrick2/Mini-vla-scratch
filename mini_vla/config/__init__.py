"""Configuration loading and validation."""

from mini_vla.config.loader import load_config
from mini_vla.config.schema import validate_config

__all__ = ["load_config", "validate_config"]
