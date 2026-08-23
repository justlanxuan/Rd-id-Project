#!/usr/bin/env python3
"""Inventory our extractor artifacts for orientation-bearing fields.

This is deliberately separate from the dataset/source orientation inventory.
It audits the artifacts actually produced by the Re-ID extraction/augmentation
pipeline under /data/lyxie and records whether orientation survives into the
canonical skeleton consumed by matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "yolopose_high",
    "alphapose",
    "fmpose3d",
    "motionagformer",
    "tcpformer",
    "wham",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.dtype.kind in "biufc" and array.size:
        finite = np.isfinite(array)
        result.update(
            {
                "finite_fraction": float(finite.mean()),
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
            }
        )
    return result


def _npz_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "suffix": path.suffix}
    try:
        with np.load(path, allow_pickle=True) as archive:
            result["keys"] = list(archive.files)
            result["arrays"] = {key: _array_summary(archive[key]) for key in archive.files}
            orientation_keys = [
                key
                for key in archive.files
                if any(token in key.lower() for token in ("orient", "rotation", "quat", "axis_angle", "pose_world"))
            ]
            result["orientation_like_keys"] = orientation_keys
            for metadata_key in ("metadata", "metadata_json"):
                if metadata_key in archive.files:
                    try:
                        result["metadata"] = json.loads(str(archive[metadata_key].item()))
                    except Exception:
                        result["metadata"] = str(archive[metadata_key].item())
                    break
            for key in ("source", "extract_method"):
                if key in archive.files:
                    result[key] = str(archive[key].item())
    except Exception as exc:  # keep failures explicit, never replace with empty result
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _scan_files(paths: list[Path]) -> dict[str, Any]:
    """Scan every artifact in a collection; keep failures explicit."""
    orientation_files = 0
    errors = 0
    nonfinite = 0
    shapes: set[tuple[int, ...]] = set()
    for path in paths:
        summary = _npz_summary(path)
        if "error" in summary:
            errors += 1
            continue
        if summary.get("orientation_like_keys"):
            orientation_files += 1
        skeleton = summary.get("arrays", {}).get("skeleton") or summary.get("arrays", {}).get("extract_skeleton")
        if skeleton:
            shapes.add(tuple(skeleton["shape"]))
            if skeleton.get("finite_fraction") != 1.0:
                nonfinite += 1
    return {
        "files": len(paths),
        "orientation_key_files": orientation_files,
        "errors": errors,
        "nonfinite_skeleton_files": nonfinite,
        "skeleton_shapes": [list(shape) for shape in sorted(shapes)],
    }


def _record(method: str, canonical_root: Path, algorithm_root: Path, raw_roots: list[Path]) -> dict[str, Any]:
    canonical_files = sorted(canonical_root.rglob("*.npz")) if canonical_root.exists() else []
    algorithm_files = sorted(algorithm_root.glob("*.npz")) if algorithm_root.exists() else []
    raw_files: list[Path] = []
    for root in raw_roots:
        if root.exists():
            raw_files.extend(sorted(root.rglob("*.npz")))
    raw_files = sorted(set(raw_files))
    sample = algorithm_files[0] if algorithm_files else (canonical_files[0] if canonical_files else None)
    canonical_sample = _npz_summary(canonical_files[0]) if canonical_files else None
    algorithm_sample = _npz_summary(algorithm_files[0]) if algorithm_files else None
    raw_sample = _npz_summary(raw_files[0]) if raw_files else None

    canonical_orientation = sorted((canonical_sample or {}).get("orientation_like_keys", []))
    algorithm_orientation = sorted((algorithm_sample or {}).get("orientation_like_keys", []))
    raw_orientation = sorted((raw_sample or {}).get("orientation_like_keys", []))
    if raw_orientation or algorithm_orientation:
        classification = "direct_orientation_raw_but_not_canonical" if not canonical_orientation else "direct_orientation_propagated"
    elif method in {"fmpose3d", "motionagformer", "tcpformer", "wham"}:
        classification = "3d_joints_derived_heading"
    else:
        classification = "2d_joints_derived_proxy"

    return {
        "method": method,
        "canonical_root": str(canonical_root),
        "algorithm_root": str(algorithm_root),
        "raw_roots": [str(root) for root in raw_roots],
        "canonical_npz_count": len(canonical_files),
        "algorithm_npz_count": len(algorithm_files),
        "raw_npz_count": len(raw_files),
        "full_scan": {
            "canonical": _scan_files(canonical_files),
            "algorithm": _scan_files(algorithm_files),
            # Raw WHAM recon files can contain 6890-vertex meshes; sample/raw
            # counts are sufficient here and avoid materializing the full mesh
            # corpus during a metadata-only inventory.
            "raw": {"files": len(raw_files), "sample_orientation_like_keys": raw_orientation},
        },
        "sample_algorithm": algorithm_sample,
        "sample_canonical": canonical_sample,
        "sample_raw": raw_sample,
        "classification": classification,
        "orientation_is_in_canonical_pair_input": bool(canonical_orientation),
        "orientation_like_keys": {
            "canonical": canonical_orientation,
            "algorithm": algorithm_orientation,
            "raw": raw_orientation,
        },
        "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest() if sample else None,
    }


def build_inventory(lyxie_root: Path) -> dict[str, Any]:
    reid = lyxie_root / "ReID"
    egohumans = reid / "Pipeline/Re-id-Project-egohumans"
    canonical_parent = egohumans / "data/skeleton_aug/S06_source_ablation"
    algorithm_parent = egohumans / "Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs"
    records = []
    for method in METHODS:
        raw_roots = []
        if method == "yolopose_high":
            raw_roots.append(egohumans / "data/custom_annotation_video_pose/yolo_pose_high")
        elif method == "alphapose":
            raw_roots.extend(
                [
                    reid / "Pipeline/Skeleton_Extractors/2D/AlphaPose/results",
                    egohumans / "Experiment/RB-Skeleton-Aug/S06_Algo_Aug/raw_outputs/alphapose",
                ]
            )
        elif method == "wham":
            raw_roots.extend(
                [
                    lyxie_root / "ReID_imu_generation/outputs/wham/recon",
                    lyxie_root / "ReID_imu_generation/outputs/wham/imu/processed_smoke",
                    reid / "Pipeline/despite/results/wham_outputs",
                ]
            )
        records.append(
            _record(
                method,
                canonical_parent / method,
                algorithm_parent / method,
                raw_roots,
            )
        )
    payload = {
        "schema_version": "g12.extractor_orientation_inventory.v1",
        "scope": "our extractor artifacts, not dataset-native orientation",
        "lyxie_root": str(lyxie_root),
        "canonical_input_contract": {
            "field": "extract_skeleton",
            "shape": "(T,N,17,3)",
            "canonical_transform": "root-centered torso-scaled H36M17 xyz (2D sources have z=0)",
        },
        "classification_rules": {
            "direct_orientation_propagated": "orientation-like field exists in canonical pair input",
            "direct_orientation_raw_but_not_canonical": "raw extractor artifact has orientation-like field, but canonical skeleton consumed by matching does not",
            "3d_joints_derived_heading": "3D joint positions permit a coordinate-dependent torso heading proxy; no direct rotation field",
            "2d_joints_derived_proxy": "2D joint positions permit only image-plane shoulder/hip proxy; no world heading",
        },
        "records": records,
    }
    return _jsonable(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lyxie-root", type=Path, default=Path("/data/lyxie"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = build_inventory(args.lyxie_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(inventory["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
