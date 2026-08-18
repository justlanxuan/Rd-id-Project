# Experiment Note: A1-source-inventory
"""Build a read-only inventory and quality summary for G9 skeleton sources.

The script never modifies source artifacts. It writes only the requested JSON
reports under the G9 external artifact root. Missing roots are recorded as
``missing`` so an unavailable backend cannot be mistaken for a successful
empty result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

H36M_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


@dataclass
class FileSummary:
    path: str
    suffix: str
    size_bytes: int
    sha256: str
    schema: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def finite_fraction(array: np.ndarray) -> float | None:
    if not np.issubdtype(array.dtype, np.number):
        return None
    if array.size == 0:
        return 1.0
    return float(np.isfinite(array).mean())


def skeleton_view(array: np.ndarray) -> np.ndarray | None:
    """Return a [T, N, 17, C] view for common canonical layouts."""
    if array.ndim == 3 and array.shape[-2] == 17:
        return array[:, None, :, :]
    if array.ndim == 4 and array.shape[-2] == 17:
        return array
    return None


def skeleton_quality(array: np.ndarray) -> dict[str, Any] | None:
    view = skeleton_view(array)
    if view is None:
        return None
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "coordinate_dim": int(view.shape[-1]),
        "finite_fraction": finite_fraction(view),
        "mean": float(np.nanmean(view)),
        "std": float(np.nanstd(view)),
        "min": float(np.nanmin(view)),
        "max": float(np.nanmax(view)),
        "frame_count": int(view.shape[0]),
        "person_count": int(view.shape[1]),
    }
    if view.shape[-1] >= 2:
        points = view[..., :3]
        lengths = []
        for left, right in H36M_EDGES:
            lengths.append(np.linalg.norm(points[:, :, left] - points[:, :, right], axis=-1))
        bone_lengths = np.stack(lengths, axis=-1)
        finite_lengths = bone_lengths[np.isfinite(bone_lengths) & (bone_lengths > 0)]
        if finite_lengths.size:
            result["bone_length_mean"] = float(finite_lengths.mean())
            result["bone_length_cv"] = float(finite_lengths.std() / max(finite_lengths.mean(), 1e-12))
    return result


def npz_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    schema: dict[str, Any] = {"keys": [], "arrays": {}}
    quality: dict[str, Any] | None = None
    numeric_fingerprint: str | None = None
    with np.load(path, allow_pickle=True) as archive:
        schema["keys"] = list(archive.files)
        for key in archive.files:
            array = archive[key]
            schema["arrays"][key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite_fraction": finite_fraction(array),
            }
            if quality is None and ("skeleton" in key or key in {"pose", "keypoints"}):
                quality = skeleton_quality(array)
            if numeric_fingerprint is None and key in {"skeleton", "gt_skeleton", "extract_skeleton"}:
                view = skeleton_view(array)
                if view is not None:
                    sample = np.ascontiguousarray(view[: min(32, len(view))])
                    numeric_fingerprint = hashlib.sha256(sample.tobytes()).hexdigest()
    return schema, quality, numeric_fingerprint


def summarize_file(
    path: Path,
    root: Path,
    include_hash: bool,
    max_npz_inspect_bytes: int,
) -> FileSummary:
    schema = None
    quality = None
    if path.suffix == ".npz" and path.stat().st_size <= max_npz_inspect_bytes:
        schema, quality, _ = npz_summary(path)
    return FileSummary(
        path=str(path.relative_to(root)),
        suffix=path.suffix,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path) if include_hash else "not_computed",
        schema=schema,
        quality=quality,
    )


def scan_source(
    name: str,
    source_type: str,
    root: Path,
    sample_limit: int,
    include_hash: bool,
    max_npz_inspect_bytes: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "source_type": source_type,
        "root": str(root),
        "status": "present" if root.exists() else "missing",
        "file_counts": {},
        "sample_files": [],
        "sample_quality": [],
    }
    if not root.exists():
        return result
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        suffix = path.suffix.lower() or "<none>"
        result["file_counts"][suffix] = result["file_counts"].get(suffix, 0) + 1
    selected = [path for path in files if path.suffix.lower() in {".npz", ".npy", ".json", ".pkl", ".pth"}]
    selected = selected[:sample_limit]
    for path in selected:
        summary = summarize_file(path, root, include_hash, max_npz_inspect_bytes)
        result["sample_files"].append(asdict(summary))
        if summary.quality:
            result["sample_quality"].append(
                {"path": summary.path, **summary.quality}
            )
    result["total_files"] = len(files)
    result["sample_limit"] = sample_limit
    return result


def configured_sources() -> list[tuple[str, str, Path]]:
    s06 = Path("/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/Experiment/RB-Skeleton-Aug/S06_Algo_Aug")
    return [
        ("totalcapture_gt", "reference_gt", Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences")),
        ("egohumans_pose2d", "2d_pose_cache", Path("/data/fzliang/reid-project/egohumans/preprocessed/g6_egohumans_source/sequences")),
        ("custom_g6_canonical", "canonical_custom", Path("/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_rawcsv7d_swapsess")),
        ("s06_alphapose", "2d_detector", s06 / "algorithm_outputs/alphapose"),
        ("s06_yolopose_high", "2d_detector", s06 / "algorithm_outputs/yolopose_high"),
        ("s06_fmpose3d", "3d_lifter", s06 / "algorithm_outputs/fmpose3d"),
        ("s06_motionagformer", "3d_lifter", s06 / "algorithm_outputs/motionagformer"),
        ("s06_tcpformer", "3d_lifter", s06 / "algorithm_outputs/tcpformer"),
        ("s06_wham", "smpl_3d", s06 / "algorithm_outputs/wham"),
        ("custom_yolopose_high_raw", "2d_detector_raw", Path("/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/data/custom_annotation_video_pose/yolo_pose_high")),
        ("alphapose_raw_results", "2d_detector_raw", Path("/data/lyxie/ReID/Pipeline/Skeleton_Extractors/2D/AlphaPose/results")),
        ("wham_raw_output2", "smpl_3d_raw", Path("/data/lyxie/ReID/Pipeline/Skeleton_Extractors/3D/WHAM/output2")),
        (
            "egohumans_pose2d_cache_sync",
            "2d_pose_cache_raw",
            Path("/data/lyxie/ReID/Data/egohumans/cache_sync_action_20_5/pose2d"),
        ),
        (
            "egohumans_pose2d_cache_w24",
            "2d_pose_cache_raw",
            Path("/data/lyxie/ReID/Data/egohumans/EgoHumans_5imu_w24/cache_action_1.2_0.8/data/pose2d"),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/fzliang/reid-project/g9/e1_gap_audit"),
    )
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument(
        "--max-npz-inspect-mb",
        type=int,
        default=128,
        help="Do not decompress NPZ samples larger than this size.",
    )
    args = parser.parse_args()
    if args.sample_limit < 1:
        raise SystemExit("--sample-limit must be positive")
    if args.max_npz_inspect_mb < 1:
        raise SystemExit("--max-npz-inspect-mb must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = [
        scan_source(
            name,
            source_type,
            root,
            args.sample_limit,
            args.full_hash,
            args.max_npz_inspect_mb * 1024 * 1024,
        )
        for name, source_type, root in configured_sources()
    ]
    fingerprint_groups: dict[str, list[str]] = {}
    for source in sources:
        for sample in source["sample_files"]:
            if sample["suffix"] != ".npz":
                continue
            arrays = sample.get("schema", {}).get("arrays", {})
            if "skeleton" not in arrays and "gt_skeleton" not in arrays and "extract_skeleton" not in arrays:
                continue
            path = Path(source["root"]) / sample["path"]
            try:
                _, _, fingerprint = npz_summary(path)
            except Exception as exc:  # preserve audit evidence, fail later in quality report
                source.setdefault("warnings", []).append(f"{path}: {exc}")
                continue
            if fingerprint:
                fingerprint_groups.setdefault(fingerprint, []).append(source["name"])
    duplicates = [names for names in fingerprint_groups.values() if len(set(names)) > 1]
    report = {
        "schema_version": "g9-e1-1",
        "generated_by": "A1_build_source_inventory.py",
        "hash_mode": "full" if args.full_hash else "sample_files_only",
        "sources": sources,
        "potential_duplicate_skeleton_fingerprints": duplicates,
        "candidate_backends_pending_smoke": [
            "PromptHMR",
            "Human3R",
            "GENMO",
            "SMPLest-X",
            "TRAM",
            "VIBE",
            "DenseWarper",
        ],
    }
    output = args.output_root / "source_inventory.json"
    output.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "sources": {item["name"]: {"status": item["status"], "total_files": item.get("total_files", 0)} for item in sources},
        "potential_duplicate_groups": duplicates,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
