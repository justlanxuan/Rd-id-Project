"""Shared helpers for writing normalized sequence-level NPZs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_sequence_npz(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def write_sequence_meta(path: Path, meta: dict[str, Any]) -> Path:
    output = dict(meta)
    output.setdefault("data_layer", "standardized")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return path
