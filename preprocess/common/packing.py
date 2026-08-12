"""Shared packing-side preprocessing helpers.

These helpers normalize unified sequence NPZ files, collect source NPZs, and
write lightweight metadata for downstream slice/train stages.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def collect_npzs(root: Path, side: str | None = None) -> list[Path]:
    direct = sorted(root.glob("*.npz"))
    if direct:
        return direct
    if side:
        token = str(side).strip().lower()
        return sorted(path for path in root.rglob("*.npz") if token in path.name.lower())
    return sorted(root.rglob("*.npz"))


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


def normalize_sequence_id(path: Path, data: dict, prefix: str = "egohumans") -> str:
    raw = scalar_string(data, "sequence_id", path.stem)
    if raw.startswith(f"{prefix}_"):
        return raw
    if raw.startswith("custom_"):
        return f"{prefix}_{raw[len('custom_'):]}"
    return f"{prefix}_{raw}"


def write_normalized_npz(src: Path, dst: Path, dataset: str = "egohumans") -> dict:
    data = dict(np.load(src, allow_pickle=True))
    sequence_id = normalize_sequence_id(src, data, prefix=dataset)
    data["dataset"] = np.array(dataset, dtype=object)
    data["sequence_id"] = np.array(sequence_id, dtype=object)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **data)

    video_path = scalar_string(data, "video_path", "")
    meta = {
        "dataset": dataset,
        "sequence_id": sequence_id,
        "source_npz": str(src),
        "video_path": video_path,
        "n_frames": int(data["frame_ids"].shape[0]) if "frame_ids" in data else None,
        "n_imu": int(data["imu_ids"].shape[0]) if "imu_ids" in data else None,
        "n_gt": int(data["gt_person_ids"].shape[0]) if "gt_person_ids" in data else None,
        "has_gt": "gt_skeleton" in data,
    }
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def copy_npz_tree(source_root: Path, output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for npz_path in sorted(source_root.rglob("*.npz")):
        dst = output_root / npz_path.name
        shutil.copy2(npz_path, dst)
        copied.append(dst)
    return copied