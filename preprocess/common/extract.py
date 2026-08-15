"""Shared video skeleton extraction runner.

Dataset modules should only discover media and write manifests. This module
owns the common detector/tracker/pose-estimator dispatch for all datasets.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# COCO joint order (17 joints)
COCO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# H36M 17 joint names (order used in the training code)
H36M_JOINTS = [
    "Hip",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "Spine",
    "Thorax",
    "Neck/Nose",
    "Head",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
]


def resolve_extract_config(config: Dict[str, Any]) -> Dict[str, Any] | None:
    """Resolve the public top-level extractor config with optional local overrides."""
    top_level = config.get("extract")
    preprocess = config.get("preprocess")
    nested = preprocess.get("extract") if isinstance(preprocess, dict) else None
    if top_level is not None and not isinstance(top_level, dict):
        raise TypeError(f"extract config must be a mapping, got {type(top_level).__name__}")
    if nested is not None and not isinstance(nested, dict):
        raise TypeError(
            f"preprocess.extract config must be a mapping, got {type(nested).__name__}"
        )
    if top_level is None and nested is None:
        return None
    return {**(top_level or {}), **(nested or {})}


def coco_to_h36m17(coco_keypoints: np.ndarray) -> np.ndarray:
    """Convert COCO 17-joint keypoints to H36M 17-joint format."""
    n_frames = coco_keypoints.shape[0]
    h36m = np.zeros((n_frames, 17, 3), dtype=np.float32)

    coco_map = {name: i for i, name in enumerate(COCO_JOINTS)}
    h36m_to_coco = {
        0: None,
        1: coco_map["right_hip"],
        2: coco_map["right_knee"],
        3: coco_map["right_ankle"],
        4: coco_map["left_hip"],
        5: coco_map["left_knee"],
        6: coco_map["left_ankle"],
        7: None,
        8: None,
        9: coco_map["nose"],
        10: None,
        11: coco_map["left_shoulder"],
        12: coco_map["left_elbow"],
        13: coco_map["left_wrist"],
        14: coco_map["right_shoulder"],
        15: coco_map["right_elbow"],
        16: coco_map["right_wrist"],
    }

    for h36m_idx, coco_idx in h36m_to_coco.items():
        if coco_idx is not None:
            h36m[:, h36m_idx, :2] = coco_keypoints[:, coco_idx, :2]
            h36m[:, h36m_idx, 2] = 0.0

    h36m[:, 0, :2] = (
        coco_keypoints[:, coco_map["left_hip"], :2] + coco_keypoints[:, coco_map["right_hip"], :2]
    ) / 2
    hips = h36m[:, 0, :2]
    shoulders = (
        coco_keypoints[:, coco_map["left_shoulder"], :2]
        + coco_keypoints[:, coco_map["right_shoulder"], :2]
    ) / 2
    h36m[:, 7, :2] = (hips + shoulders) / 2
    h36m[:, 8, :2] = shoulders
    nose = coco_keypoints[:, coco_map["nose"], :2]
    neck = (shoulders + nose) / 2
    head_offset = nose - neck
    h36m[:, 10, :2] = nose + head_offset * 0.5
    return h36m


def _frame_num_from_image_id(image_id: str) -> int:
    stem = Path(image_id).stem
    try:
        return int(stem)
    except ValueError:
        return 0


def load_alphapose_multiperson(skeleton_json: Path) -> tuple[dict[int, list[dict]], list[int]]:
    """Load AlphaPose skeleton JSON supporting multiple tracks per frame."""
    with open(skeleton_json) as f:
        data = json.load(f)

    frames: dict[int, list[dict]] = {}
    track_ids_set: set[int] = set()

    for entry in data:
        image_id = entry.get("image_id", "")
        frame_idx = _frame_num_from_image_id(image_id)

        keypoints_flat = entry.get("keypoints", [])
        if len(keypoints_flat) < 17 * 3:
            continue

        coco_kpts = np.zeros((1, 17, 3), dtype=np.float32)
        for j in range(17):
            coco_kpts[0, j, 0] = keypoints_flat[j * 3]
            coco_kpts[0, j, 1] = keypoints_flat[j * 3 + 1]
            coco_kpts[0, j, 2] = keypoints_flat[j * 3 + 2]

        h36m_kpts = coco_to_h36m17(coco_kpts)[0]

        box = entry.get("box", [0.0, 0.0, 0.0, 0.0])
        if len(box) < 4:
            box = [0.0, 0.0, 0.0, 0.0]
        x, y, w, h = box
        bbox = np.array([float(x), float(y), float(x + w), float(y + h)], dtype=np.float32)

        raw_idx = entry.get("idx", 0)
        if isinstance(raw_idx, list):
            track_id_list = [int(x) for x in raw_idx]
        else:
            track_id_list = [int(raw_idx)]

        for track_id in track_id_list:
            track_ids_set.add(track_id)
            det = {
                "track_id": track_id,
                "bbox": bbox,
                "keypoints": h36m_kpts,
                "score": float(entry.get("score", 0.0)),
            }
            frames.setdefault(frame_idx, []).append(det)

    return frames, sorted(track_ids_set)


def validate_skeleton_artifact(skeleton_json: str | Path) -> Path:
    """Validate a reusable AlphaPose-format skeleton artifact."""
    path = Path(skeleton_json).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Skeleton artifact not found: {path}")
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid skeleton JSON {path}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Skeleton artifact is empty: {path}")
    valid_entries = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        keypoints = entry.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) < 17 * 3:
            continue
        values = np.asarray(keypoints[: 17 * 3], dtype=np.float32)
        if np.isfinite(values).all():
            valid_entries += 1
    if valid_entries == 0:
        raise ValueError(f"Skeleton artifact has no valid 17-joint detections: {path}")
    return path


def run_extraction_if_enabled(
    video_path: str | Path | None,
    output_dir: str | Path | None,
    extract_cfg: Dict[str, Any] | None,
) -> Path | None:
    """Run skeleton extraction when explicitly enabled.

    The helper is intentionally thin and extensible: it accepts a config dict
    and can dispatch to any extractor implementation supported by the
    ``src.modules.extractors`` package, while staying optional for datasets that
    do not have video data or do not want to extract on every run.
    """

    if not extract_cfg:
        return None
    if not isinstance(extract_cfg, dict):
        raise TypeError(f"extract config must be a mapping, got {type(extract_cfg).__name__}")
    if extract_cfg.get("enabled") is False:
        return None

    resolved_video = Path(video_path).expanduser().resolve() if video_path is not None else None
    resolved_output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if resolved_video is None or not resolved_video.is_file():
        raise FileNotFoundError(f"Extraction video not found: {resolved_video}")
    if resolved_output is None:
        resolved_output = resolved_video.parent / f"{resolved_video.stem}_extract"

    resolved_output.mkdir(parents=True, exist_ok=True)
    from src.modules.extractors import assemble_extract_config

    local_cfg = assemble_extract_config(extract_cfg)
    local_cfg["results_root"] = str(resolved_output.parent)
    extractor = build_video_skeleton_extractor(local_cfg)
    skeleton_json = process_video_skeleton(
        resolved_video,
        extractor,
        local_cfg,
        dry_run=False,
        result_name=resolved_output.name,
    )
    return Path(skeleton_json)


def build_video_skeleton_extractor(extract_cfg: Dict[str, Any]) -> Any:
    detector = extract_cfg.get("detector")
    tracker = extract_cfg.get("tracker")
    pose_estimator = extract_cfg.get("pose_estimator")

    if detector == "alphapose" and tracker == "alphapose" and pose_estimator == "alphapose":
        extractor_name = "alphapose_full"
    elif detector in (None, "bytetrack") and tracker == "bytetrack" and pose_estimator == "alphapose":
        extractor_name = "bytetrack_alphapose"
    elif pose_estimator == "wham" and detector is None and tracker is None:
        extractor_name = "wham"
    else:
        raise ValueError(
            f"Unsupported extractor combination: detector={detector}, tracker={tracker}, "
            f"pose_estimator={pose_estimator}. Supported: (alphapose+alphapose+alphapose), "
            f"(bytetrack+bytetrack+alphapose), or (wham)."
        )

    from src.modules.extractors import build_extractor

    return build_extractor(
        extractor_name,
        extract_cfg,
        allow_experimental=bool(extract_cfg.get("allow_experimental", False)),
    )


def _needs_merge(skeleton_json: Path, cfg: Dict[str, Any]) -> bool:
    merge = cfg.get("merge_tracklets", {})
    if not merge.get("enabled", False):
        return False
    unmerged_json = skeleton_json.with_name("skeleton_unmerged.json")
    return skeleton_json.exists() and not unmerged_json.exists()


def _run_merge(json_path: Path, cfg: Dict[str, Any], dry_run: bool) -> None:
    cmd, merged_json = _build_merge_cmd(json_path, cfg)
    print(f"\n[CMD] {' '.join(cmd)}")
    if dry_run:
        print("[DRY_RUN] skip merge")
        return
    _execute_cmd(cmd)
    if not merged_json.exists():
        raise FileNotFoundError(f"Merged JSON was not produced: {merged_json}")
    unmerged_json = json_path.with_name("skeleton_unmerged.json")
    shutil.move(str(json_path), str(unmerged_json))
    shutil.move(str(merged_json), str(json_path))
    print(f"[MERGE] Saved unmerged JSON to {unmerged_json}")
    print(f"[MERGE] Overwrote skeleton JSON with merged IDs: {json_path}")


def _build_merge_cmd(json_path: Path, cfg: Dict[str, Any]) -> tuple[list[str], Path]:
    """Compose the merge_tracklets command and return (cmd, merged_json_path)."""
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "src" / "utils" / "merge_tracklets.py"
    if not script.exists():
        raise FileNotFoundError(f"merge_tracklets script not found: {script}")

    merge = cfg.get("merge_tracklets", {})
    merged_json = json_path.with_name("skeleton_merged.json")
    cmd = [
        sys.executable,
        str(script),
        "--json_path",
        str(json_path),
        "--output_json",
        str(merged_json),
        "--max_gap",
        str(merge.get("max_gap", 10000000)),
        "--score_thresh",
        str(merge.get("score_thresh", 2.2)),
        "--max_norm_dist",
        str(merge.get("max_norm_dist", 2.8)),
        "--max_size_diff",
        str(merge.get("max_size_diff", 1.8)),
    ]
    if merge.get("fill_gaps", False):
        cmd.append("--fill_gaps")
    known = merge.get("known_num_people")
    if known is not None:
        cmd.extend(["--known_num_people", str(known)])
    return cmd, merged_json


def _execute_cmd(cmd: list[str]) -> None:
    """Execute a shell command using subprocess and raise on non-zero exit."""
    subprocess.run(cmd, check=True)


def process_video_skeleton(
    video_path: Path,
    extractor: Any,
    extract_cfg: Dict[str, Any],
    dry_run: bool = False,
    result_name: str | None = None,
) -> Path:
    results_root = Path(extract_cfg.get(
        "results_root",
        "/data/fzliang/reid-project/extract_outputs",
    )).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    video_name = result_name or video_path.stem
    video_result_dir = results_root / video_name
    video_result_dir.mkdir(parents=True, exist_ok=True)

    skeleton_json = video_result_dir / "skeleton.json"
    summary_path = video_result_dir / "pipeline_run_summary.json"
    fingerprint_payload = {
        "video": str(video_path.expanduser().resolve()),
        "extractor": f"{type(extractor).__module__}.{type(extractor).__qualname__}",
        "config": {
            key: value
            for key, value in extract_cfg.items()
            if key not in {"force", "invalid_cache_policy", "limit", "results_root", "reuse_existing", "skip_existing"}
        },
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    reuse_existing = bool(extract_cfg.get("reuse_existing", extract_cfg.get("skip_existing", True)))
    force = bool(extract_cfg.get("force", False))
    invalid_cache_policy = str(extract_cfg.get("invalid_cache_policy", "error")).strip().lower()
    if invalid_cache_policy not in {"error", "reextract"}:
        raise ValueError(f"Unsupported invalid_cache_policy={invalid_cache_policy!r}; use 'error' or 'reextract'.")
    existing_skeleton = skeleton_json.exists()
    merge_needed = _needs_merge(skeleton_json, extract_cfg)
    cache_status = "missing"

    if reuse_existing and existing_skeleton and not force:
        try:
            validate_skeleton_artifact(skeleton_json)
        except (FileNotFoundError, ValueError):
            cache_status = "invalid"
            if invalid_cache_policy == "error":
                raise
        else:
            cache_status = "reused"
            if not merge_needed:
                print(f"[REUSE] Validated existing skeleton: {skeleton_json}")

    if cache_status == "reused":
        print(f"[REUSE-EXTRACT] Reusing existing {skeleton_json} before merge")
    elif dry_run:
        print(f"[DRY_RUN] Would extract {video_path} -> {video_result_dir}")
    else:
        skeleton_json = Path(extractor.extract(str(video_path), str(video_result_dir)))
        validate_skeleton_artifact(skeleton_json)
        cache_status = "reextracted" if existing_skeleton else "extracted"

    if merge_needed:
        _run_merge(skeleton_json, extract_cfg, dry_run)

    summary = {
        "video": str(video_path),
        "video_name": video_name,
        "video_result_dir": str(video_result_dir),
        "skeleton_json": str(skeleton_json),
        "cache_status": cache_status,
        "provenance_status": "verified_current_run" if cache_status in {"extracted", "reextracted"} else "adopted_existing",
        "config_fingerprint": config_fingerprint,
        "extractor": fingerprint_payload["extractor"],
        "reuse_existing": reuse_existing,
        "force": force,
        "merge_enabled": extract_cfg.get("merge_tracklets", {}).get("enabled", False),
    }
    if not dry_run:
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
    print(f"\nPipeline finished for {video_name}")
    print(f"Skeleton JSON: {skeleton_json}")
    return skeleton_json
