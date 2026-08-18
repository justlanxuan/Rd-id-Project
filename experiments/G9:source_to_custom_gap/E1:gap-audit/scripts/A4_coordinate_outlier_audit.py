# Experiment Note: A4-coordinate-outlier-audit
"""Audit raw versus root/torso-normalized skeleton coordinates and S06 outliers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TOTAL_ROOT = Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences")
EGO_ROOT = Path("/data/fzliang/reid-project/egohumans/preprocessed/g6_egohumans_source/sequences")
CUSTOM_ROOT = Path("/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_rawcsv7d_swapsess")
S06_ROOT = Path("/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs")
METHODS = ("alphapose", "yolopose_high", "fmpose3d", "motionagformer", "tcpformer", "wham")


def representation(skeleton: np.ndarray, visibility: np.ndarray | None) -> str:
    if skeleton.shape[-1] == 2:
        return "2d_xy"
    if visibility is not None and np.allclose(skeleton[..., 2], 0.0):
        return "2d_xy_zero_z"
    if np.all((skeleton[..., 2] >= 0) & (skeleton[..., 2] <= 1)):
        return "2d_xy_visibility"
    return "3d_xyz"


def norm_coords(skeleton: np.ndarray, visibility: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Root center and scale by pelvis-thorax distance, retaining dimensions."""
    coords = skeleton[..., :2] if representation(skeleton, visibility).startswith("2d_") else skeleton
    centered = coords - coords[..., 0:1, :]
    torso = np.linalg.norm(centered[..., 8, :] - centered[..., 0, :], axis=-1)
    valid_torso = np.isfinite(torso) & (torso > 1e-6)
    scale = np.where(valid_torso, torso, 1.0)
    return centered / scale[..., None, None], valid_torso


def quantile(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "q01": None, "q50": None, "q99": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "q01": float(np.quantile(array, 0.01)),
        "q50": float(np.quantile(array, 0.50)),
        "q99": float(np.quantile(array, 0.99)),
    }


def load(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=True) as archive:
        key = "skeleton" if "skeleton" in archive else "gt_skeleton"
        skeleton = np.asarray(archive[key], dtype=np.float64)
        visibility = np.asarray(archive["visibility"]) if "visibility" in archive else None
    if skeleton.ndim == 3:
        skeleton = skeleton[:, None]
    return skeleton, visibility


def paths_for(source: str) -> list[Path]:
    if source == "totalcapture_gt":
        return sorted(TOTAL_ROOT.glob("*.npz"))
    if source == "egohumans_canonical":
        return sorted(EGO_ROOT.glob("*.npz"))
    if source == "custom_canonical":
        return sorted(CUSTOM_ROOT.glob("fold*/sequences/*.npz"))
    return sorted((S06_ROOT / source.removeprefix("s06_")).glob("*.npz"))


def source_summary(source: str, max_files: int | None) -> dict[str, Any]:
    paths = paths_for(source)
    if max_files is not None:
        paths = paths[:max_files]
    raw_values: list[float] = []
    norm_values: list[float] = []
    torso_values: list[float] = []
    representation_counts: dict[str, int] = defaultdict(int)
    extreme = 0
    total = 0
    invalid_scale = 0
    extreme_examples: list[dict[str, Any]] = []
    extreme_by_joint: dict[str, int] = defaultdict(int)
    files_with_extreme = 0
    for path in paths:
        try:
            skeleton, visibility = load(path)
        except (OSError, ValueError, EOFError):
            continue
        name = representation(skeleton, visibility)
        representation_counts[name] += 1
        coords = skeleton[..., :2] if name.startswith("2d_") else skeleton
        finite = np.isfinite(coords)
        values = coords[finite]
        raw_values.extend(values.reshape(-1).tolist())
        extreme += int(np.sum(np.abs(values) > 10.0))
        total += int(values.size)
        raw_coords = coords
        extreme_indices = np.argwhere(np.abs(raw_coords) > 10.0)
        if extreme_indices.size:
            files_with_extreme += 1
            for index in extreme_indices:
                person, joint, dimension = [int(item) for item in index[-3:]]
                extreme_by_joint[f"{joint}:{dimension}"] += 1
                if len(extreme_examples) < 20:
                    extreme_examples.append(
                        {
                            "path": str(path),
                            "frame": int(index[0]),
                            "person": person,
                            "joint": joint,
                            "dimension": dimension,
                            "value": float(raw_coords[tuple(index)]),
                        }
                    )
        normalized, valid_scale = norm_coords(skeleton, visibility)
        invalid_scale += int(np.sum(~valid_scale))
        norm_values.extend(normalized[np.isfinite(normalized)].reshape(-1).tolist())
        torso = np.linalg.norm(coords[..., 8, :] - coords[..., 0, :], axis=-1)
        torso_values.extend(torso[np.isfinite(torso)].reshape(-1).tolist())
    return {
        "files": len(paths),
        "representation_counts": dict(sorted(representation_counts.items())),
        "raw_coordinate_quantiles": quantile(raw_values),
        "root_torso_normalized_quantiles": quantile(norm_values),
        "torso_scale_quantiles": quantile(torso_values),
        "raw_abs_gt_10_fraction": extreme / total if total else None,
        "raw_abs_gt_10_count": extreme,
        "files_with_raw_abs_gt_10": files_with_extreme,
        "raw_abs_gt_10_by_joint_dimension": dict(sorted(extreme_by_joint.items())),
        "raw_abs_gt_10_examples": sorted(extreme_examples, key=lambda item: -abs(item["value"])),
        "invalid_or_zero_torso_scale_count": invalid_scale,
    }


def normalized_pairwise() -> dict[str, Any]:
    pairs: dict[str, list[float]] = defaultdict(list)
    methods = list(METHODS)
    for left_index, left_method in enumerate(methods):
        for right_method in methods[left_index + 1 :]:
            left_root = S06_ROOT / left_method
            right_root = S06_ROOT / right_method
            key = f"{left_method}__{right_method}"
            for left_path in sorted(left_root.glob("*.npz")):
                right_path = right_root / left_path.name
                if not right_path.exists():
                    continue
                left, left_vis = load(left_path)
                right, right_vis = load(right_path)
                if left.shape != right.shape:
                    continue
                if representation(left, left_vis)[:2] != representation(right, right_vis)[:2]:
                    continue
                left_n, left_valid = norm_coords(left, left_vis)
                right_n, right_valid = norm_coords(right, right_vis)
                valid = left_valid & right_valid
                if not np.any(valid):
                    continue
                delta = np.abs(left_n - right_n)[valid]
                pairs[key].extend(delta[np.isfinite(delta)].reshape(-1).tolist())
    return {method: quantile(values) for method, values in sorted(pairs.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/fzliang/reid-project/g9/e1_gap_audit"))
    parser.add_argument("--max-files", type=int, default=None, help="optional smoke limit per source")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = [
        "totalcapture_gt",
        "egohumans_canonical",
        "custom_canonical",
        *[f"s06_{method}" for method in METHODS],
    ]
    report = {
        "schema_version": "g9-e1-coordinate-outlier-1",
        "raw_extreme_threshold": "absolute coordinate > 10 in source units; diagnostic only, not a universal rejection threshold",
        "normalization": "subtract pelvis joint 0 and divide by pelvis-thorax distance (joint 8); 2D and 3D remain separate representation tracks",
        "sources": {source: source_summary(source, args.max_files) for source in sources},
        "s06_alpha_normalized_pairwise": normalized_pairwise(),
        "limitations": [
            "The root/torso normalization is a controlled comparison, not a claim that the source's original coordinate contract is correct.",
            "Raw_abs_gt_10 is an outlier locator; visual and source-frame review are required before exclusion.",
        ],
    }
    output = args.output_root / "coordinate_outlier_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "sources": list(report["sources"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
