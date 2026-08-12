"""Extractor-domain public API and component-fragment assembly."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from src.config.config import substitute_variables
from src.modules.extractors.alphapose_full import AlphaPoseFullExtractor
from src.modules.extractors.base import ExtractorCapabilities, VideoSkeletonExtractor
from src.modules.extractors.bytetrack_alphapose import ByteTrackAlphaPoseExtractor
from src.modules.extractors.registry import EXTRACTOR_REGISTRY, build_extractor
from src.modules.extractors.wham import WHAMExtractor

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_fragment(
    category: str,
    name: str,
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load a component fragment, not a complete workflow configuration."""
    path = REPO_ROOT / "configs" / category / f"{name}.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Extractor fragment must be a mapping: {path}")
    substitutions = {
        key: value
        for key, value in os.environ.items()
        if isinstance(value, str)
    }
    substitutions.update(variables or {})
    resolved = substitute_variables(payload, substitutions)
    unresolved = sorted(
        {
            match
            for value in _walk_strings(resolved)
            for match in ENVIRONMENT_VARIABLE_PATTERN.findall(value)
        }
    )
    if unresolved:
        names = ", ".join(unresolved)
        raise ValueError(
            f"Extractor fragment {path} requires environment variable(s): {names}"
        )
    return resolved


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def assemble_extract_config(extract_cfg: dict[str, Any]) -> dict[str, Any]:
    variables: dict[str, str] = {"repo_root": str(REPO_ROOT)}
    for key, value in extract_cfg.items():
        if key.endswith("_root") and isinstance(value, str):
            variables[key] = value

    merged: dict[str, Any] = {}
    for component in ("detector", "tracker", "pose_estimator"):
        name = extract_cfg.get(component)
        if name:
            merged = _deep_merge(
                merged,
                load_fragment(f"{component}s", str(name), variables=variables),
            )
    return substitute_variables(_deep_merge(merged, extract_cfg), variables)


__all__ = [
    "AlphaPoseFullExtractor",
    "ByteTrackAlphaPoseExtractor",
    "EXTRACTOR_REGISTRY",
    "ExtractorCapabilities",
    "VideoSkeletonExtractor",
    "WHAMExtractor",
    "assemble_extract_config",
    "build_extractor",
    "load_fragment",
]
