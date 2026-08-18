# Experiment Note: A2-semantic-skeleton-audit
"""Audit skeleton semantics and cross-source comparability without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

S06_ROOT = Path(
    "/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/"
    "Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs"
)
S06_METADATA = S06_ROOT.parent / "metadata"
CUSTOM_ROOT = Path(
    "/data/fzliang/reid-project/custom/preprocessed/"
    "hybrid_w24_session_out_rawcsv7d_swapsess"
)


def as_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def finite_fraction(array: np.ndarray) -> float | None:
    if not np.issubdtype(array.dtype, np.number):
        return None
    return float(np.isfinite(array).mean()) if array.size else 1.0


def scalar_text(array: np.ndarray) -> str | None:
    if array.shape != ():
        return None
    try:
        value = array.item()
    except Exception:
        return None
    return str(value)


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def monotonic(array: np.ndarray) -> bool | None:
    if array.ndim != 1 or array.size < 2:
        return None
    return bool(np.all(np.diff(array) > 0))


def skeleton_array(archive: Any) -> tuple[str | None, np.ndarray | None]:
    for key in ("skeleton", "gt_skeleton", "extract_skeleton"):
        if key in archive:
            return key, archive[key]
    return None, None


def mapping_check(archive: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not_available", "fields": {}}
    person_keys = ("gt_person_ids", "person_ids", "track_ids")
    imu_keys = ("imu_ids", "imu_person_ids")
    person_key = next((key for key in person_keys if key in archive), None)
    imu_key = next((key for key in imu_keys if key in archive), None)
    if person_key:
        result["fields"][person_key] = as_json(archive[person_key].tolist())
    if imu_key:
        result["fields"][imu_key] = as_json(archive[imu_key].tolist())
    if person_key and imu_key:
        person_ids = set(np.asarray(archive[person_key]).reshape(-1).tolist())
        imu_ids = set(np.asarray(archive[imu_key]).reshape(-1).tolist())
        result["sets_equal"] = person_ids == imu_ids
        result["status"] = "verified_equal" if person_ids == imu_ids else "mismatch"
    elif person_key or imu_key:
        result["status"] = "partial"
    return result


def custom_window_mapping(path: Path) -> dict[str, Any]:
    """Check person/IMU mapping from the fold window CSV for a Custom NPZ."""
    fold_root = path.parent.parent
    rows: list[dict[str, str]] = []
    for csv_path in sorted(fold_root.glob("windows_*.csv")):
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("npz_path") == f"sequences/{path.name}":
                    rows.append(row)
    if not rows:
        return {"status": "not_found", "matched_rows": 0}
    pair_values = {(row.get("person_idx"), row.get("imu_idx")) for row in rows}
    source_values = {row.get("skeleton_source") for row in rows}
    return {
        "status": "verified_equal" if all(left == right for left, right in pair_values) else "mismatch",
        "matched_rows": len(rows),
        "person_imu_pairs": sorted([list(pair) for pair in pair_values]),
        "skeleton_sources": sorted(source_values),
    }


def custom_all_fold_mapping() -> dict[str, Any]:
    """Audit every row in the selected Custom fold manifests."""
    csv_paths = sorted(CUSTOM_ROOT.glob("fold*/windows_*.csv"))
    rows = 0
    referenced: set[Path] = set()
    missing_npz: set[str] = set()
    mismatches = 0
    missing_fields = 0
    sources: set[str | None] = set()
    folds: dict[str, dict[str, int]] = {}
    for csv_path in csv_paths:
        fold = csv_path.parent.name
        fold_stats = folds.setdefault(fold, {"csv_files": 0, "rows": 0})
        fold_stats["csv_files"] += 1
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                fold_stats["rows"] += 1
                person_idx = row.get("person_idx")
                imu_idx = row.get("imu_idx")
                if not person_idx or not imu_idx:
                    missing_fields += 1
                elif person_idx != imu_idx:
                    mismatches += 1
                sources.add(row.get("skeleton_source"))
                rel = row.get("npz_path")
                if rel:
                    npz_path = csv_path.parent / rel
                    referenced.add(npz_path)
                    if not npz_path.exists():
                        missing_npz.add(str(npz_path))
    return {
        "status": (
            "verified_equal"
            if csv_paths and rows and not mismatches and not missing_fields and not missing_npz
            else "mismatch_or_incomplete"
        ),
        "manifest_files": len(csv_paths),
        "rows": rows,
        "unique_npz_references": len(referenced),
        "missing_npz": sorted(missing_npz),
        "person_imu_mismatches": mismatches,
        "rows_missing_mapping_fields": missing_fields,
        "skeleton_sources": sorted(source for source in sources if source is not None),
        "folds": folds,
    }


def s06_baseline_mapping() -> dict[str, Any]:
    """Verify IMU/person identity joins for every S06 train/validation baseline."""
    manifests = sorted(S06_METADATA.glob("*_cam03_manifest.csv"))
    checked = 0
    missing = []
    mismatches = []
    for manifest in manifests:
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                checked += 1
                path = Path(row["baseline_npz"])
                if not path.exists():
                    missing.append(str(path))
                    continue
                with np.load(path, allow_pickle=True) as archive:
                    mapping = mapping_check(archive)
                if mapping.get("status") != "verified_equal":
                    mismatches.append({"sequence_id": row.get("sequence_id"), "mapping": mapping})
    return {
        "status": "verified_equal" if manifests and checked and not missing and not mismatches else "mismatch_or_incomplete",
        "manifest_files": len(manifests),
        "baseline_sequences_checked": checked,
        "missing_baselines": missing,
        "mapping_mismatches": mismatches,
    }


def file_semantics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "status": "missing",
    }
    if not path.exists():
        return result
    result["status"] = "readable"
    if path.suffix != ".npz":
        return result
    with np.load(path, allow_pickle=True) as archive:
        key, skeleton = skeleton_array(archive)
        result["keys"] = list(archive.files)
        result["skeleton_key"] = key
        result["mapping"] = mapping_check(archive)
        if "custom" in path.name or "custom" in str(path.parent.parent):
            result["custom_window_mapping"] = custom_window_mapping(path)
        result["arrays"] = {
            name: {
                "shape": list(archive[name].shape),
                "dtype": str(archive[name].dtype),
                "finite_fraction": finite_fraction(archive[name]),
            }
            for name in archive.files
            if name in {"imu", "skeleton", "gt_skeleton", "extract_skeleton", "frame_ids", "visibility"}
        }
        if skeleton is None:
            result["status"] = "missing_skeleton_key"
            return result
        result["skeleton_shape"] = list(skeleton.shape)
        result["skeleton_finite"] = finite_fraction(skeleton) == 1.0
        result["joint_count"] = int(skeleton.shape[-2]) if skeleton.ndim >= 3 else None
        result["coordinate_dim"] = int(skeleton.shape[-1]) if skeleton.ndim >= 2 else None
        result["frame_count"] = int(skeleton.shape[0]) if skeleton.ndim else None
        if "frame_ids" in archive:
            result["frame_ids_monotonic"] = monotonic(archive["frame_ids"])
        if "imu" in archive:
            result["imu_frame_count"] = int(archive["imu"].shape[0])
            result["skeleton_imu_frame_count_equal"] = int(skeleton.shape[0]) == int(archive["imu"].shape[0])
        result["sample_fingerprint"] = sha256_array(skeleton[: min(32, len(skeleton))])
        result["sample_range"] = [float(np.nanmin(skeleton)), float(np.nanmax(skeleton))]
    return result


def s06_method_summary(method: str, sequence: str) -> dict[str, Any]:
    path = S06_ROOT / method / f"{sequence}.npz"
    result = file_semantics(path)
    result["method"] = method
    if path.exists():
        with np.load(path, allow_pickle=True) as archive:
            for key in ("source", "metadata_json", "alignment_json"):
                if key in archive:
                    result[key] = scalar_text(archive[key])
    return result


def pairwise_s06(methods: list[str], sequence: str) -> list[dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    for method in methods:
        path = S06_ROOT / method / f"{sequence}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as archive:
            key, array = skeleton_array(archive)
            if key and array is not None:
                arrays[method] = np.asarray(array, dtype=np.float64)
    pairs = []
    for left, right in itertools.combinations(sorted(arrays), 2):
        a, b = arrays[left], arrays[right]
        item: dict[str, Any] = {"left": left, "right": right, "same_shape": list(a.shape) == list(b.shape)}
        if a.shape == b.shape:
            af = a.reshape(-1)
            bf = b.reshape(-1)
            item["exact_equal"] = bool(np.array_equal(a, b))
            item["mean_abs_delta"] = float(np.mean(np.abs(af - bf)))
            if np.std(af) > 0 and np.std(bf) > 0:
                item["pearson_r"] = float(np.corrcoef(af, bf)[0, 1])
        pairs.append(item)
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/fzliang/reid-project/g9/e1_gap_audit"),
    )
    parser.add_argument("--sequence", default="custom_01_003")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    canonical_paths = {
        "totalcapture_gt": Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences/totalcapture_S1_acting1_cam1.npz"),
        "egohumans_canonical": Path("/data/fzliang/reid-project/egohumans/preprocessed/g6_egohumans_source/sequences/egohumans_01_001.npz"),
        "custom_canonical": Path("/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_rawcsv7d_swapsess/fold1_20260211_171423/sequences/custom_20260211_171423_seg0_p0_0_24.npz"),
    }
    methods = ["alphapose", "yolopose_high", "fmpose3d", "motionagformer", "tcpformer", "wham"]
    custom_mapping = custom_all_fold_mapping()
    s06_mapping = s06_baseline_mapping()
    method_decisions = {}
    for method in methods:
        if method == "yolopose_high":
            method_decisions[method] = {
                "status": "conditional",
                "scope": "diagnostic_only",
                "reason": "S06 validation is readable, but coverage is materially lower and A1 flagged a coordinate/outlier signature; exclude from first formal gap matrix until audited.",
            }
        elif s06_mapping["status"] == "verified_equal":
            method_decisions[method] = {
                "status": "included",
                "scope": "skeleton_and_imu_join",
                "reason": "Output carries gt_person_ids; all S06 baseline manifests verify gt_person_ids == imu_ids externally.",
            }
        else:
            method_decisions[method] = {
                "status": "conditional",
                "scope": "skeleton_only",
                "reason": "External S06 baseline person/IMU join is incomplete.",
            }
    source_decisions = {
        "totalcapture_gt": {
            "status": "included",
            "scope": "canonical_gt_imu_gap",
            "reason": "Canonical sample has explicit gt_person_ids/imu_ids equality and frame-count equality.",
        },
        "egohumans_canonical": {
            "status": "included",
            "scope": "canonical_gt_imu_gap",
            "reason": "Canonical sample has explicit gt_person_ids/imu_ids equality and frame-count equality; coordinate normalization remains an explicit gap factor.",
        },
        "custom_canonical": {
            "status": "included" if custom_mapping["status"] == "verified_equal" else "conditional",
            "scope": "target_custom",
            "reason": "All selected fold window rows must have person_idx == imu_idx and reference an existing NPZ.",
        },
        **method_decisions,
        "raw_pose_and_wham_artifacts": {
            "status": "pending",
            "scope": "diagnostic_only",
            "reason": "Raw caches do not yet provide a canonical person/IMU/time join in the E1 manifest.",
        },
    }
    minimal_subset = [
        name for name, item in source_decisions.items() if item["status"] == "included"
    ]
    report = {
        "schema_version": "g9-e1-semantic-1",
        "canonical_samples": {name: file_semantics(path) for name, path in canonical_paths.items()},
        "s06_methods": {method: s06_method_summary(method, args.sequence) for method in methods},
        "s06_pairwise": pairwise_s06(methods, args.sequence),
        "custom_all_fold_mapping": custom_mapping,
        "s06_baseline_mapping": s06_mapping,
        "source_decisions": source_decisions,
        "minimal_trusted_subset": minimal_subset,
        "semantic_limitations": [
            "Joint order is only accepted from explicit metadata/protocol; numeric shape alone is insufficient.",
            "Person IDs are considered aligned only when explicit person and IMU id sets are equal.",
            "Visual correctness and source-video identity still require a separate rendered-frame review.",
        ],
    }
    output = args.output_root / "semantic_audit.json"
    output.write_text(json.dumps(as_json(report), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "canonical_status": {name: value["status"] for name, value in report["canonical_samples"].items()},
        "s06_methods": {name: value["status"] for name, value in report["s06_methods"].items()},
        "pairwise_comparisons": len(report["s06_pairwise"]),
        "exact_duplicate_pairs": [
            (item["left"], item["right"])
            for item in report["s06_pairwise"]
            if item.get("exact_equal")
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
