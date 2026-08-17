"""Full-session FrameAcc for Custom data with unmerged tracklets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.datasets.custom_session import CustomTrackletSession, load_custom_tracklet_session
from src.engine.similarity import encode_hybrid_precomputed, model_similarity_matrix
from src.models.checkpoint import load_model_checkpoint
from src.modules.encoders.hybrid import imu_sequence_features, raw_pose_sequence, skeleton_tokens
from src.modules.matchers.tracklet_temporal import (
    build_temporal_tracklet_matcher,
    temporal_matcher_metadata,
)


def _window_starts(t_len: int, window_size: int, stride: int) -> list[int]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if t_len < window_size:
        return []
    starts = list(range(0, t_len - window_size + 1, stride))
    final_start = t_len - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _frame_counts(
    session: CustomTrackletSession,
    frame_assignments: np.ndarray,
) -> tuple[float, int, int]:
    correct = 0
    total = 0
    for time_index in range(len(session.frame_ids)):
        for gt_index, gt_id in enumerate(session.gt_person_ids):
            if not session.gt_visibility[time_index, gt_index]:
                continue
            expected_track = int(session.gt_to_extract_map[time_index, gt_index])
            if expected_track < 0:
                continue
            total += 1
            matched_imu = np.flatnonzero(frame_assignments[time_index] == expected_track)
            if len(matched_imu) == 1 and int(session.imu_ids[int(matched_imu[0])]) == int(gt_id):
                correct += 1
    return (float(correct / total) if total else 0.0, int(correct), int(total))


def _nearest_center_assignments(
    t_len: int,
    n_imu: int,
    centers: list[int],
    assignments: list[np.ndarray],
) -> np.ndarray:
    frame_assignments = np.full((t_len, n_imu), -1, dtype=np.int64)
    if not centers:
        return frame_assignments
    centers_array = np.asarray(centers, dtype=np.int64)
    for time_index in range(t_len):
        nearest = int(np.argmin(np.abs(centers_array - time_index)))
        frame_assignments[time_index] = assignments[nearest]
    return frame_assignments


def _global_assignment(relative: np.ndarray, active: np.ndarray) -> np.ndarray:
    result = np.full(relative.shape[0], -1, dtype=np.int64)
    matched = relative >= 0
    result[matched] = active[relative[matched]]
    return result


def _instantaneous_assignment(similarity: np.ndarray, active: np.ndarray) -> np.ndarray:
    relative = np.full(similarity.shape[0], -1, dtype=np.int64)
    if similarity.shape[0] and similarity.shape[1]:
        rows, columns = linear_sum_assignment(-similarity)
        relative[rows] = columns
    return _global_assignment(relative, active)


def _load_model(cfg, checkpoint: Path, device: torch.device):
    from src.engine.common import build_alignment_model_from_cfg

    model, model_name = build_alignment_model_from_cfg(cfg, device)
    report = load_model_checkpoint(model, model_name, checkpoint, strict=False)
    if report.missing_keys or report.unexpected_keys:
        print(
            f"[WARN] checkpoint loaded with missing={len(report.missing_keys)}, "
            f"unexpected={len(report.unexpected_keys)}"
        )
    if not model.capabilities.segment_frame_acc:
        raise RuntimeError(f"Model {model_name!r} does not support sequence FrameAcc.")
    model.eval()
    return model


def _evaluate_one_session(
    cfg,
    model,
    session: CustomTrackletSession,
    device: torch.device,
) -> dict[str, Any]:
    frame_cfg = cfg.TEST.METRICS.FRAME_ACC
    window_size = int(frame_cfg.WINDOW_SIZE)
    stride = int(frame_cfg.STRIDE)
    per_window_features = bool(frame_cfg.PER_WINDOW_FEATURES)
    matcher = build_temporal_tracklet_matcher(frame_cfg)

    t_len = len(session.frame_ids)
    pose2d = session.extract_skeleton[..., :2].astype(np.float32)
    imu7 = session.imu.astype(np.float32)
    n_imu = int(imu7.shape[1])
    skel_smooth = int(getattr(model.video_encoder, "skeleton_smooth_kernel", 9))
    image_height = float(getattr(model.video_encoder, "image_height", 1080.0))
    image_width = float(getattr(model.video_encoder, "image_width", 1920.0))
    imu_smooth = int(getattr(model.imu_encoder, "imu_smooth_kernel", 5))
    imu_feature_mode = str(getattr(model.imu_encoder, "feature_mode", "raw"))
    if not per_window_features:
        with torch.no_grad():
            pose_full = torch.from_numpy(pose2d.transpose(1, 0, 2, 3)).float()
            imu_full = torch.from_numpy(imu7.transpose(1, 0, 2)).float()
            raw_full = raw_pose_sequence(pose_full, skel_smooth, image_height, image_width)
            vec_full = skeleton_tokens(pose_full, skel_smooth, image_height, image_width)
            imu_feat_full = imu_sequence_features(imu_full, imu_smooth, imu_feature_mode)

    centers: list[int] = []
    history_assignments: list[np.ndarray] = []
    instant_assignments: list[np.ndarray] = []
    window_predictions: list[dict[str, Any]] = []
    for start in _window_starts(t_len, window_size, stride):
        end = start + window_size
        active = np.flatnonzero(session.extract_visibility[start:end].any(axis=0))
        if active.size == 0:
            continue
        with torch.no_grad():
            if per_window_features:
                skeleton = torch.from_numpy(pose2d[start:end, active].transpose(1, 0, 2, 3)).float().to(device)
                imu = torch.from_numpy(imu7[start:end].transpose(1, 0, 2)).float().to(device)
                if getattr(model, "cross_pair_head", None) is not None:
                    similarity = model.cross_pair_logits(imu, skeleton).detach().cpu().numpy()
                else:
                    output = model(imu=imu, skeleton=skeleton)
                    similarity = model_similarity_matrix(
                        model,
                        F.normalize(output["imu"], dim=-1),
                        F.normalize(output["video"], dim=-1),
                        cosine_weight=float(frame_cfg.COSINE_WEIGHT),
                        pair_logit_weight=float(frame_cfg.PAIR_LOGIT_WEIGHT),
                    )
            else:
                imu_emb, video_emb = encode_hybrid_precomputed(
                    model,
                    raw_full[active, start:end].to(device),
                    vec_full[active, start:end].to(device),
                    imu_feat_full[:n_imu, start:end].to(device),
                )
                similarity = model_similarity_matrix(
                    model,
                    imu_emb,
                    video_emb,
                    cosine_weight=float(frame_cfg.COSINE_WEIGHT),
                    pair_logit_weight=float(frame_cfg.PAIR_LOGIT_WEIGHT),
                )

        labels = tuple(session.tracklet_labels[index] for index in active)
        temporal_result = matcher.update(similarity, labels)
        history_global = _global_assignment(temporal_result.assignment, active)
        instant_global = _instantaneous_assignment(similarity, active)
        center = (start + end) // 2
        centers.append(center)
        history_assignments.append(history_global)
        instant_assignments.append(instant_global)
        window_predictions.append(
            {
                "start": int(start),
                "end": int(end),
                "center": int(center),
                "active_tracklet_indices": active.astype(np.int64).tolist(),
                "active_tracklet_labels": list(labels),
                "similarity": np.asarray(similarity, dtype=np.float64).tolist(),
                **temporal_result.prediction_fields(),
                "instantaneous_imu_to_tracklet": instant_global.tolist(),
                "history_imu_to_tracklet": history_global.tolist(),
            }
        )

    history_frames = _nearest_center_assignments(t_len, n_imu, centers, history_assignments)
    instant_frames = _nearest_center_assignments(t_len, n_imu, centers, instant_assignments)
    history_acc, history_correct, total = _frame_counts(session, history_frames)
    instant_acc, instant_correct, instant_total = _frame_counts(session, instant_frames)
    if instant_total != total:
        raise RuntimeError("History and instantaneous baselines produced different denominators")
    return {
        "sequence_id": session.sequence_id,
        "T": int(t_len),
        "frame_ids": session.frame_ids.tolist(),
        "imu_ids": session.imu_ids.tolist(),
        "tracklet_labels": list(session.tracklet_labels),
        "num_tracklets": int(len(session.tracklet_labels)),
        "max_simultaneous_tracklets": int(session.extract_visibility.sum(axis=1).max(initial=0)),
        "correct": history_correct,
        "total": total,
        "frame_acc": history_acc,
        "instantaneous_correct": instant_correct,
        "instantaneous_frame_acc": instant_acc,
        "window_predictions": window_predictions,
        "frame_assignments": history_frames.tolist(),
        "instantaneous_frame_assignments": instant_frames.tolist(),
    }


def evaluate_full_session_frameacc(
    cfg,
    checkpoint: Path,
    device: torch.device,
    *,
    resolve_path,
) -> dict[str, Any]:
    """Evaluate whole Custom sessions without constructing segment NPZs."""
    frame_cfg = cfg.TEST.METRICS.FRAME_ACC
    session_root_value = str(frame_cfg.SESSION_ROOT).strip()
    tracklet_root_value = str(frame_cfg.TRACKLET_ROOT).strip()
    if not session_root_value or not tracklet_root_value:
        raise ValueError("Full-session FrameAcc requires SESSION_ROOT and TRACKLET_ROOT")
    session_root = resolve_path(session_root_value)
    tracklet_root = resolve_path(tracklet_root_value)
    imu_root_value = str(frame_cfg.CUSTOM_IMU_ROOT).strip()
    imu_root = resolve_path(imu_root_value) if imu_root_value else None
    sessions = [str(value) for value in frame_cfg.SESSIONS]
    if not sessions:
        sessions = [str(value) for value in cfg.SLICE.TEST_SESSIONS]
    if not sessions:
        raise ValueError("Full-session FrameAcc requires FRAME_ACC.SESSIONS or SLICE.TEST_SESSIONS")
    swap_sessions = {str(value) for value in frame_cfg.CUSTOM_IMU_RAW_SWAP_SESSIONS}
    filename = str(frame_cfg.TRACKLET_FILENAME)
    model = _load_model(cfg, checkpoint, device)

    session_results: list[dict[str, Any]] = []
    for session_name in sessions:
        aligned_path = session_root / f"custom_{session_name}.npz"
        tracklet_path = tracklet_root / session_name / filename
        session = load_custom_tracklet_session(
            aligned_path,
            tracklet_path,
            custom_imu_root=imu_root,
            raw_swap=session_name in swap_sessions,
            normalize_extract_skeleton=bool(frame_cfg.NORMALIZE_EXTRACT_SKELETON),
        )
        session_results.append(_evaluate_one_session(cfg, model, session, device))

    per_session = {
        str(result["sequence_id"]).removeprefix("custom_"): {
            key: result[key]
            for key in (
                "num_tracklets",
                "max_simultaneous_tracklets",
                "correct",
                "total",
                "frame_acc",
                "instantaneous_correct",
                "instantaneous_frame_acc",
            )
        }
        for result in session_results
    }
    correct = sum(int(result["correct"]) for result in session_results)
    instant_correct = sum(int(result["instantaneous_correct"]) for result in session_results)
    total = sum(int(result["total"]) for result in session_results)
    return {
        "method": "frame_acc",
        "prediction_schema_version": "1.2",
        "mode": "full_sessions_unmerged_tracklets",
        "session_root": str(session_root),
        "tracklet_root": str(tracklet_root),
        "tracklet_filename": filename,
        "custom_imu_root": "" if imu_root is None else str(imu_root),
        "custom_imu_raw_swap_sessions": sorted(swap_sessions),
        "sessions": sessions,
        "window_size": int(frame_cfg.WINDOW_SIZE),
        "stride": int(frame_cfg.STRIDE),
        "per_window_features": bool(frame_cfg.PER_WINDOW_FEATURES),
        "temporal_matcher": temporal_matcher_metadata(frame_cfg),
        "per_session": per_session,
        "session_results": session_results,
        "correct": int(correct),
        "total": int(total),
        "frame_acc": (
            float(np.mean([float(result["frame_acc"]) for result in session_results]))
            if session_results
            else 0.0
        ),
        "weighted_frame_acc": float(correct / total) if total else 0.0,
        "instantaneous_correct": int(instant_correct),
        "instantaneous_weighted_frame_acc": float(instant_correct / total) if total else 0.0,
    }


__all__ = ["evaluate_full_session_frameacc"]
