"""Preprocess-facing helpers backed by the single official config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.config import load_config as _load_config


def load_config(
    config_path: str | Path | None,
    extra_variables: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    if not config_path:
        return {}
    return _load_config(config_path, extra_variables=extra_variables)


def load_section_config(config_path: str | Path | None, section: str | None = None) -> Dict[str, Any]:
    data = load_config(config_path)
    if not section:
        return data
    value = data.get(section)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section {section!r} in config must be a mapping")
    return value


def resolve_config(config_path: str | Path | None) -> Dict[str, Any]:
    return load_config(config_path)


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default
