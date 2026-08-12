"""Shared helpers for writing normalized sequence-level NPZs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def scalar_string(data: dict, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    try:
        if getattr(value, "shape", None) == ():
            return str(value.item())
    except Exception:
        pass
    return str(value)


def write_sequence_npz(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def write_sequence_meta(path: Path, meta: dict[str, Any]) -> Path:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
