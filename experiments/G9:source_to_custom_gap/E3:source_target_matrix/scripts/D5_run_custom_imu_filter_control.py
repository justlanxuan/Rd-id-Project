# Experiment Note: D5-custom-imu-filter-control
"""Compare embedded Custom 7D IMU with conservative quaternion filters.

The control keeps the held-out Custom tracklets, GT/person order, window/stride,
matcher and fixed EgoHumans checkpoint unchanged.  Only the embedded IMU
quaternion stream is changed: one variant fills only invalid frames, while a
second also unit-normalizes every quaternion. Acceleration and all
skeleton/visibility fields are preserved. This is a diagnostic fusion control,
not a claim that filtering is the correct production policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

ALIGNED_ROOT = Path(
    "/data/fzliang/reid-project/custom/preprocessed/"
    "custom_hybrid_finetune_from_egohumans/aligned_sequences"
)
TRACKLET_ROOT = Path("/data/fzliang/reid-project/custom/skeleton/alphapose")
CHECKPOINT = Path(
    "/data/fzliang/reid-project/g6/c9a5d3099979296a72314eba66274855e03ab1eb/"
    "artifacts/train/train__source__egohumans__seed0/best.pt"
)
SESSIONS = ("20260211_171423", "20260211_171724", "20260211_172257", "20260211_172522")


def _unit_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, 1e-12)


def filter_quaternions(
    imu: np.ndarray, *, normalize_valid: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """Nearest-valid-fill invalid frames, optionally normalizing all quaternions."""
    output = np.asarray(imu, dtype=np.float32).copy()
    quaternion = output[..., 3:7]
    norms = np.linalg.norm(quaternion, axis=-1)
    valid = np.isfinite(norms) & (norms >= 0.9) & (norms <= 1.1)
    valid &= np.isfinite(quaternion).all(axis=-1)
    for person in range(quaternion.shape[1]):
        person_valid = valid[:, person]
        if person_valid.any():
            indices = np.flatnonzero(person_valid)
            for frame in range(quaternion.shape[0]):
                source = frame if person_valid[frame] else int(indices[np.argmin(np.abs(indices - frame))])
                quaternion[frame, person] = quaternion[source, person]
        else:
            quaternion[:, person] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if normalize_valid:
            quaternion[:, person] = _unit_quaternion(quaternion[:, person])
    output[..., 3:7] = quaternion
    report = {
        "frames": int(norms.size),
        "invalid_frames": int((~valid).sum()),
        "invalid_fraction": float((~valid).mean()) if norms.size else 0.0,
        "frames_replaced": int((~valid).sum()),
        "valid_frames_renormalized": int(valid.sum()) if normalize_valid else 0,
        "post_filter_norm_min": float(np.linalg.norm(quaternion, axis=-1).min()) if norms.size else None,
        "post_filter_norm_max": float(np.linalg.norm(quaternion, axis=-1).max()) if norms.size else None,
        "policy": (
            "valid=[0.9,1.1] unchanged; invalid=nearest-valid quaternion; no valid=identity"
            if not normalize_valid
            else "valid=[0.9,1.1] then unit-normalize; invalid=nearest-valid quaternion; no valid=identity"
        ),
    }
    return output, report


def prepare_filtered(
    aligned_root: Path,
    output_root: Path,
    sessions: list[str],
    *,
    normalize_valid: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for session in sessions:
        source_path = aligned_root / f"custom_{session}.npz"
        destination = output_root / source_path.name
        with np.load(source_path, allow_pickle=True) as archive:
            payload = {key: archive[key] for key in archive.files}
            filtered, report = filter_quaternions(
                np.asarray(payload["imu"], dtype=np.float32), normalize_valid=normalize_valid
            )
            payload["imu"] = filtered
        np.savez_compressed(destination, **payload)
        reports[session] = {"source": str(source_path), "output": str(destination), **report}
    return reports


def evaluate_variant(
    aligned_root: Path,
    tracklet_root: Path,
    sessions: list[str],
    device: str,
) -> dict[str, Any]:
    import torch

    from src.config import load_cfg
    from src.engine.session_frameacc import evaluate_full_session_frameacc

    cfg = load_cfg("configs/evaluation/custom_full_session_egohumans_pretrained.yaml")
    cfg.defrost()
    frame_cfg = cfg.TEST.METRICS.FRAME_ACC
    frame_cfg.MODE = "full_session"
    frame_cfg.SESSION_ROOT = str(aligned_root)
    frame_cfg.TRACKLET_ROOT = str(tracklet_root)
    frame_cfg.TRACKLET_FILENAME = "skeleton_unmerged.json"
    frame_cfg.CUSTOM_IMU_ROOT = ""
    frame_cfg.CUSTOM_IMU_RAW_SWAP_SESSIONS = ()
    frame_cfg.SESSIONS = tuple(sessions)
    frame_cfg.WINDOW_SIZE = 24
    frame_cfg.STRIDE = 16
    frame_cfg.PER_WINDOW_FEATURES = True
    frame_cfg.NORMALIZE_EXTRACT_SKELETON = True
    frame_cfg.ENABLED = True
    cfg.TRAIN.DEVICE = device
    cfg.TEST.DEVICE = device
    cfg.freeze()
    result = evaluate_full_session_frameacc(
        cfg,
        CHECKPOINT,
        device=torch.device(device),
        resolve_path=lambda value: Path(value).expanduser().resolve(),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-root", type=Path, default=ALIGNED_ROOT)
    parser.add_argument("--tracklet-root", type=Path, default=TRACKLET_ROOT)
    parser.add_argument(
        "--filtered-root",
        type=Path,
        default=Path("/data/fzliang/reid-project/g9/e3_source_target/custom_imu_invalid_fill"),
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("/data/fzliang/reid-project/g9/e3_source_target/custom_imu_unit_normalized"),
    )
    parser.add_argument("--output", type=Path, default=Path("/data/fzliang/reid-project/g9/e3_source_target/custom_imu_filter_control.json"))
    parser.add_argument("--sessions", nargs="*", default=list(SESSIONS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sessions = list(args.sessions)
    filtered_reports = prepare_filtered(
        args.aligned_root, args.filtered_root, sessions, normalize_valid=False
    )
    normalized_reports = prepare_filtered(
        args.aligned_root, args.normalized_root, sessions, normalize_valid=True
    )
    raw = evaluate_variant(args.aligned_root, args.tracklet_root, sessions, args.device)
    filtered = evaluate_variant(args.filtered_root, args.tracklet_root, sessions, args.device)
    normalized = evaluate_variant(args.normalized_root, args.tracklet_root, sessions, args.device)
    report = {
        "schema_version": "g9-e3-custom-imu-filter-control-1",
        "protocol": {
            "checkpoint": str(CHECKPOINT),
            "window_size": 24,
            "stride": 16,
            "sessions": sessions,
            "tracklet_root": str(args.tracklet_root),
            "same_skeleton_and_gt": True,
        },
        "filter": filtered_reports,
        "unit_normalization": normalized_reports,
        "raw": raw,
        "invalid_fill_only": filtered,
        "unit_normalized": normalized,
        "delta_invalid_fill_only": {
            "history_frame_acc": float(filtered["weighted_frame_acc"] - raw["weighted_frame_acc"]),
            "instantaneous_frame_acc": float(
                filtered["instantaneous_weighted_frame_acc"] - raw["instantaneous_weighted_frame_acc"]
            ),
            "correct": int(filtered["correct"] - raw["correct"]),
            "total": int(filtered["total"] - raw["total"]),
        },
        "delta_unit_normalized": {
            "history_frame_acc": float(normalized["weighted_frame_acc"] - raw["weighted_frame_acc"]),
            "instantaneous_frame_acc": float(
                normalized["instantaneous_weighted_frame_acc"] - raw["instantaneous_weighted_frame_acc"]
            ),
            "correct": int(normalized["correct"] - raw["correct"]),
            "total": int(normalized["total"] - raw["total"]),
        },
        "interpretation": {
            "causal_scope": "fixed-checkpoint Custom-target IMU intervention; no retraining",
            "denominator": "raw and filtered use identical visible GT frame denominator",
            "warning": "nearest-valid fill is a diagnostic policy, not a validated sensor repair",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "raw": report["raw"]["weighted_frame_acc"],
                "invalid_fill_only": report["invalid_fill_only"]["weighted_frame_acc"],
                "unit_normalized": report["unit_normalized"]["weighted_frame_acc"],
                "delta_invalid_fill_only": report["delta_invalid_fill_only"],
                "delta_unit_normalized": report["delta_unit_normalized"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
