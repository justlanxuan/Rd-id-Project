"""Unified evaluation entrypoint.

The official test surface is intentionally limited to two methods:

- FrameAcc: synchronous per-window assignment accuracy.
- Group Test: sampled group matching accuracy over sequence chunks.

Legacy evaluators remain in the repository for reproducibility, but the
pipeline should route test runs through this module.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.config import load_cfg
from src.preprocess.datasets.custom import (
    legacy_imu48_sensor_to_7d,
    load_custom_rawcsv_7d_sequence,
    load_custom_split_7d_sequence,
)
from src.datasets import WindowAlignmentDataset
from src.modules.encoders.hybrid import imu_sequence_features, raw_pose_sequence, skeleton_tokens


def parse_group_sizes(spec) -> List[int]:
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


def pair_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return -1.0
    sim = cosine_matrix(a[:n], b[:n])
    return float(np.mean(np.diag(sim)))


def model_similarity_matrix(
    model,
    imu_emb: torch.Tensor,
    video_emb: torch.Tensor,
    cosine_weight: float = 0.0,
    pair_logit_weight: float = 1.0,
) -> np.ndarray:
    cosine = imu_emb @ video_emb.t()
    if getattr(model, "pair_head", None) is None:
        return cosine.detach().cpu().numpy()
    pair_logits = model.pair_logits(imu_emb, video_emb)
    return (float(cosine_weight) * cosine + float(pair_logit_weight) * pair_logits).detach().cpu().numpy()


@dataclass
class EmbeddingBundle:
    rows: List[Dict[str, str]]
    imu: np.ndarray
    video: np.ndarray


def compute_embeddings(cfg, checkpoint: Path, device: torch.device) -> EmbeddingBundle:
    T = cfg.TRAIN
    P = cfg.PATHS
    TEST = cfg.TEST
    IMU_PRE = cfg.PREPROCESS.IMU
    is_hybrid = str(T.MODEL.TYPE).lower() == "hybrid"

    imu_mean = None
    imu_std = None
    imu_stats_json = "" if is_hybrid else _resolve_imu_stats(cfg, checkpoint)
    if (not is_hybrid) and imu_stats_json:
        stats = json.loads(Path(imu_stats_json).read_text())
        imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(stats["imu_std"], dtype=np.float32)

    imu_sensor = T.IMU_SENSOR.strip() if T.IMU_SENSOR else None
    dataset = WindowAlignmentDataset(
        P.TEST_CSV,
        root_dir=P.DATA_ROOT,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=imu_sensor,
        repeat_single_sensor=T.REPEAT_SINGLE_SENSOR,
        imu_lowpass_cutoff_hz=IMU_PRE.LOWPASS_CUTOFF_HZ,
        imu_lowpass_fs_hz=IMU_PRE.LOWPASS_FS_HZ,
        return_root_trajectory=False,
        root_source="auto",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(TEST.BATCH_SIZE or T.BATCH_SIZE),
        shuffle=False,
        num_workers=int(TEST.NUM_WORKERS or T.NUM_WORKERS),
        pin_memory=True,
    )

    from src.engine.common import build_alignment_model_from_cfg, _adapt_legacy_hybrid_state_dict

    model, _ = build_alignment_model_from_cfg(cfg, device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = _adapt_legacy_hybrid_state_dict(state)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("stats"), dict):
        stats = ckpt["stats"]
        stat_map = {
            "raw_mu": "video_encoder.raw_mu",
            "raw_sd": "video_encoder.raw_sd",
            "vec_mu": "video_encoder.vec_mu",
            "vec_sd": "video_encoder.vec_sd",
            "imu_mu": "imu_encoder.imu_mu",
            "imu_sd": "imu_encoder.imu_sd",
        }
        for src_key, dst_key in stat_map.items():
            if src_key in stats:
                state[dst_key] = torch.as_tensor(stats[src_key], dtype=torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] checkpoint loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    imu_all: List[np.ndarray] = []
    video_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            forward_kwargs = {
                "imu": batch["imu"].to(device),
                "skeleton": batch["skeleton"].to(device),
            }
            if "root_trajectory" in batch:
                forward_kwargs["root_trajectory"] = batch["root_trajectory"].to(device)
            out = model(**forward_kwargs)
            imu_all.append(F.normalize(out["imu"], dim=-1).detach().cpu().numpy())
            video_all.append(F.normalize(out["video"], dim=-1).detach().cpu().numpy())

    rows = read_csv_rows(P.TEST_CSV)
    imu = np.concatenate(imu_all, axis=0)
    video = np.concatenate(video_all, axis=0)
    if len(rows) != len(imu):
        raise ValueError(f"CSV rows and embeddings mismatch: rows={len(rows)}, embeddings={len(imu)}")
    return EmbeddingBundle(rows=rows, imu=imu, video=video)


def _resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return ((base or repo_root()) / p).resolve()


def _resolve_checkpoint(cfg, cli_checkpoint: str = "") -> Path:
    run_name = str(cfg.TEST.OUTPUT.RUN_NAME or cfg.TRAIN.OUTPUT.RUN_NAME).strip()

    checkpoint = str(cli_checkpoint or cfg.TEST.CHECKPOINT or "").strip()
    if checkpoint:
        return _resolve_path(checkpoint)
    if not run_name:
        raise ValueError("Cannot resolve checkpoint: test.output.run_name and train.output.run_name are empty.")
    output_root = _resolve_path(str(cfg.TRAIN.OUTPUT.OUTPUT_ROOT or repo_root() / "train"))
    return output_root / run_name / "best.pt"


def _resolve_imu_stats(cfg, checkpoint: Path) -> str:
    override = str(cfg.TEST.IMU_STATS_JSON or cfg.TRAIN.IMU_STATS_JSON or "").strip()
    if override:
        p = _resolve_path(override)
        if p.exists():
            return str(p)
        print(f"[WARN] test.imu_stats_json not found: {p}; falling back to checkpoint sibling.")
    sibling = checkpoint.parent / "imu_stats.json"
    return str(sibling) if sibling.exists() else ""


class FrameAccEvaluator:
    """Per-window assignment accuracy shared by all datasets.

    Rows with the same `(npz_path, session, window_start, window_end)` form one
    candidate set. Each row represents an aligned IMU/person pair, so the
    correct assignment is the diagonal after deterministic row sorting. The IMU
    side is shuffled before Hungarian matching by default to avoid identity
    tie-breaking shortcuts.
    """

    def __init__(self, shuffle_match: bool = True, seed: int = 42) -> None:
        self.shuffle_match = shuffle_match
        self.seed = seed

    @staticmethod
    def _group_key(row: Dict[str, str]) -> tuple[str, str, int, int]:
        return (
            str(row.get("npz_path", "")),
            str(row.get("session", "")),
            int(row["window_start"]),
            int(row["window_end"]),
        )

    @staticmethod
    def _sort_key(item: tuple[int, Dict[str, str]]) -> tuple[int, int, str]:
        _, row = item
        return (
            int(row.get("imu_idx", 0)),
            int(row.get("person_idx", 0)),
            str(row.get("subject", "")),
        )

    def evaluate(self, bundle: EmbeddingBundle) -> Dict[str, object]:
        grouped: Dict[tuple[str, str, int, int], List[tuple[int, Dict[str, str]]]] = {}
        for idx, row in enumerate(bundle.rows):
            grouped.setdefault(self._group_key(row), []).append((idx, row))

        rng = np.random.default_rng(self.seed)
        total = 0
        correct = 0
        window_accs: List[float] = []
        singleton = 0
        evaluated = 0

        for key in sorted(grouped):
            items = sorted(grouped[key], key=self._sort_key)
            if len(items) < 2:
                # A one-candidate window is a valid but non-discriminative
                # FrameAcc case. Count it explicitly instead of silently
                # dropping single-person datasets from the metric.
                singleton += 1
                correct += 1
                total += 1
                window_accs.append(1.0)
                evaluated += 1
                continue

            indices = np.asarray([idx for idx, _ in items], dtype=np.int64)
            sim = cosine_matrix(bundle.imu[indices], bundle.video[indices])
            if self.shuffle_match:
                perm = rng.permutation(len(indices))
                sim_for_match = sim[perm]
            else:
                perm = np.arange(len(indices))
                sim_for_match = sim

            row_ind, col_ind = linear_sum_assignment(-sim_for_match)
            n_correct = int(np.sum(perm[row_ind] == col_ind))
            n_total = int(len(indices))
            correct += n_correct
            total += n_total
            window_accs.append(float(n_correct / max(n_total, 1)))
            evaluated += 1

        return {
            "method": "frame_acc",
            "num_candidate_windows": int(len(grouped)),
            "num_evaluated_windows": int(evaluated),
            "num_singleton_windows": int(singleton),
            "num_assignments": int(total),
            "correct_assignments": int(correct),
            "frame_acc": float(correct / max(total, 1)),
            "mean_window_acc": float(np.mean(window_accs)) if window_accs else 0.0,
            "std_window_acc": float(np.std(window_accs)) if window_accs else 0.0,
            "shuffle_match": bool(self.shuffle_match),
        }


def load_skeleton_json(path: Path, tlen: int) -> np.ndarray:
    entries = json.loads(path.read_text())
    track_ids = sorted({int(e["idx"]) for e in entries})
    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    pose2d = np.zeros((tlen, len(track_ids), 17, 2), dtype=np.float32)
    for e in entries:
        frame = int(str(e["image_id"]).split(".")[0])
        if 0 <= frame < tlen:
            kpts = np.asarray(e["keypoints"], dtype=np.float32).reshape(17, 3)
            pose2d[frame, tid_to_idx[int(e["idx"])], :, :2] = kpts[:, :2]
    return pose2d


def compute_segment_frameacc_counts(npz_data, frame_assignments: np.ndarray) -> tuple[float, int, int]:
    t_len = int(npz_data["frame_ids"].shape[0])
    n_gt = int(npz_data["gt_person_ids"].shape[0])
    gt_to_extract_map = npz_data["gt_to_extract_map"]
    gt_visibility = npz_data["gt_visibility"]
    gt_person_ids = npz_data["gt_person_ids"]
    imu_ids = npz_data["imu_ids"]
    correct = 0
    total = 0
    for t in range(t_len):
        for g in range(n_gt):
            if not gt_visibility[t, g]:
                continue
            p_gt = gt_to_extract_map[t, g]
            if p_gt == -1:
                continue
            total += 1
            matched = np.where(frame_assignments[t] == p_gt)[0]
            if len(matched) == 1 and int(imu_ids[int(matched[0])]) == int(gt_person_ids[g]):
                correct += 1
    return float(correct / total) if total else 0.0, int(correct), int(total)


def load_segment_eval_inputs(
    npz_path: Path,
    custom_imu_root: Path | None,
    custom_imu_split_mode: str = "test",
    raw_swap_sessions: set[str] | None = None,
) -> tuple[object, str, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    sequence_id = str(data["sequence_id"].item())
    t_len = int(data["frame_ids"].shape[0])
    # Use the skeleton stored in the segment NPZ. It is already aligned to the
    # segment-local frame_ids and extract_person_ids used by gt_to_extract_map.
    # Re-reading extract_source JSON here can silently change track order and,
    # for nonzero-offset segments, can select the wrong temporal slice.
    pose2d = data["extract_skeleton"][:, :, :, :2].astype(np.float32)

    if custom_imu_root is None:
        imu = data["imu"].astype(np.float32)
        if imu.shape[-1] >= 7:
            if imu.shape[-1] >= 48:
                return data, sequence_id, pose2d, legacy_imu48_sensor_to_7d(imu, "L_LowArm")
            return data, sequence_id, pose2d, imu[..., :7]
        raise ValueError(f"Segment IMU has {imu.shape[-1]} channels and no custom 7D IMU root was provided.")

    session = sequence_id.split("_seg", 1)[0].split("custom_", 1)[1]
    seg_idx = int(sequence_id.rsplit("_seg", 1)[1])
    imu_list = []
    n_tracks = int(data["extract_person_ids"].shape[0])
    split_mode = custom_imu_split_mode.lower().strip()
    for p in range(n_tracks):
        if split_mode in {"rawcsv", "rawcsv_swap"}:
            imu_person_map = None
            if "imu_person_map" in data.files:
                imu_person_map = str(data["imu_person_map"].item())
            raw = load_custom_rawcsv_7d_sequence(
                custom_imu_root,
                session,
                data["frame_ids"].astype(np.int64),
                imu_person_map=imu_person_map,
                n_persons=n_tracks,
            )
            if split_mode == "rawcsv_swap" or (raw_swap_sessions and session in raw_swap_sessions):
                raw = raw[:, ::-1].copy()
            return data, sequence_id, pose2d, raw
        if split_mode == "full":
            try:
                raw = load_custom_split_7d_sequence(custom_imu_root, session, seg_idx, p, target_len=t_len)
            except FileNotFoundError:
                imu_list.append(np.zeros((t_len, 7), dtype=np.float32))
                continue
        elif split_mode == "test":
            test_npy = custom_imu_root / f"{session}_seg{seg_idx}_person{p}_test.npy"
            if not test_npy.exists():
                imu_list.append(np.zeros((t_len, 7), dtype=np.float32))
                continue
            raw = np.load(test_npy, allow_pickle=True).item()["imu"].astype(np.float32)
            if raw.ndim == 3:
                raw = raw[:, 0, :]
            pad = t_len - raw.shape[0]
            if pad > 0:
                raw = np.concatenate([np.zeros((pad, raw.shape[1]), dtype=np.float32), raw], axis=0)
        else:
            raise ValueError(
                f"Unsupported CUSTOM_IMU_SPLIT_MODE={custom_imu_split_mode!r}; "
                "use 'test', 'full', 'rawcsv', or 'rawcsv_swap'."
            )
        if raw.shape[0] < t_len:
            imu_list.append(np.zeros((t_len, 7), dtype=np.float32))
            continue
        imu_list.append(raw[:t_len, :7])
    return data, sequence_id, pose2d, np.stack(imu_list, axis=1)


def evaluate_segment_frameacc(cfg, checkpoint: Path, device: torch.device) -> Dict[str, object]:
    frame_cfg = cfg.TEST.METRICS.FRAME_ACC
    segment_root = _resolve_path(str(frame_cfg.SEGMENT_ROOT))
    custom_imu_root = str(frame_cfg.CUSTOM_IMU_ROOT).strip()
    custom_imu_path = _resolve_path(custom_imu_root) if custom_imu_root else None
    custom_imu_split_mode = str(getattr(frame_cfg, "CUSTOM_IMU_SPLIT_MODE", "test"))
    raw_swap_sessions = {str(x) for x in getattr(frame_cfg, "CUSTOM_IMU_RAW_SWAP_SESSIONS", [])}
    sessions = [str(x) for x in frame_cfg.SESSIONS]
    if not sessions:
        sessions = [str(x) for x in cfg.SLICE.TEST_SESSIONS]
    if not sessions:
        raise ValueError("FrameAcc segment mode requires TEST.METRICS.FRAME_ACC.SESSIONS or SLICE.TEST_SESSIONS.")

    from src.engine.common import build_alignment_model_from_cfg, _adapt_legacy_hybrid_state_dict

    model, _ = build_alignment_model_from_cfg(cfg, device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = _adapt_legacy_hybrid_state_dict(state)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("stats"), dict):
        stats = ckpt["stats"]
        stat_map = {
            "raw_mu": "video_encoder.raw_mu",
            "raw_sd": "video_encoder.raw_sd",
            "vec_mu": "video_encoder.vec_mu",
            "vec_sd": "video_encoder.vec_sd",
            "imu_mu": "imu_encoder.imu_mu",
            "imu_sd": "imu_encoder.imu_sd",
        }
        for src_key, dst_key in stat_map.items():
            if src_key in stats:
                state[dst_key] = torch.as_tensor(stats[src_key], dtype=torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] checkpoint loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    window_size = int(frame_cfg.WINDOW_SIZE)
    stride = int(frame_cfg.STRIDE)
    per_window_features = bool(getattr(frame_cfg, "PER_WINDOW_FEATURES", False))
    cosine_weight = float(getattr(frame_cfg, "COSINE_WEIGHT", 0.0))
    pair_logit_weight = float(getattr(frame_cfg, "PAIR_LOGIT_WEIGHT", 1.0))
    clips = []
    total_correct = 0
    total = 0
    for session in sessions:
        for path in sorted(segment_root.glob(f"custom_{session}_seg*.npz")):
            data, sequence_id, pose2d, imu7 = load_segment_eval_inputs(
                path,
                custom_imu_path,
                custom_imu_split_mode,
                raw_swap_sessions=raw_swap_sessions,
            )
            t_len = int(data["frame_ids"].shape[0])
            n_imu = int(data["imu_ids"].shape[0])
            visibility = data["extract_visibility"]
            skel_smooth = int(getattr(model.video_encoder, "skeleton_smooth_kernel", 9))
            image_height = float(getattr(model.video_encoder, "image_height", 1080.0))
            image_width = float(getattr(model.video_encoder, "image_width", 1920.0))
            imu_smooth = int(getattr(model.imu_encoder, "imu_smooth_kernel", 5))
            imu_feature_mode = str(getattr(model.imu_encoder, "feature_mode", "raw"))
            if not per_window_features:
                with torch.no_grad():
                    pose_full = torch.from_numpy(pose2d[:t_len].transpose(1, 0, 2, 3)).float()
                    imu_full = torch.from_numpy(imu7[:t_len].transpose(1, 0, 2)).float()
                    raw_full = raw_pose_sequence(pose_full, skel_smooth, image_height, image_width)
                    vec_full = skeleton_tokens(pose_full, skel_smooth, image_height, image_width)
                    imu_feat_full = imu_sequence_features(imu_full, imu_smooth, imu_feature_mode)
            centers = []
            assignments = []
            for start in range(0, t_len - window_size + 1, stride):
                end = start + window_size
                active = np.where(visibility[start:end].any(axis=0))[0]
                if active.size == 0:
                    continue
                with torch.no_grad():
                    if per_window_features:
                        sk_batch = np.stack([pose2d[start:end, p] for p in active], axis=0)
                        imu_batch = np.stack([imu7[start:end, i] for i in range(n_imu)], axis=0)
                        imu_tensor = torch.from_numpy(imu_batch).float().to(device)
                        sk_tensor = torch.from_numpy(sk_batch).float().to(device)
                        if getattr(model, "cross_pair_head", None) is not None:
                            sim = model.cross_pair_logits(imu_tensor, sk_tensor).detach().cpu().numpy()
                        else:
                            out = model(imu=imu_tensor, skeleton=sk_tensor)
                            imu_emb = F.normalize(out["imu"], dim=-1)
                            video_emb = F.normalize(out["video"], dim=-1)
                            sim = model_similarity_matrix(
                                model,
                                imu_emb,
                                video_emb,
                                cosine_weight=cosine_weight,
                                pair_logit_weight=pair_logit_weight,
                            )
                    else:
                        imu_emb, video_emb = _encode_hybrid_precomputed(
                            model,
                            raw_full[active, start:end].to(device),
                            vec_full[active, start:end].to(device),
                            imu_feat_full[:n_imu, start:end].to(device),
                        )
                        sim = model_similarity_matrix(
                            model,
                            imu_emb,
                            video_emb,
                            cosine_weight=cosine_weight,
                            pair_logit_weight=pair_logit_weight,
                        )
                row_ind, col_ind = linear_sum_assignment(-sim)
                assign = np.full(n_imu, -1, dtype=np.int64)
                for r, c in zip(row_ind, col_ind):
                    assign[r] = int(active[c])
                centers.append((start + end) // 2)
                assignments.append(assign)

            frame_assignments = np.full((t_len, n_imu), -1, dtype=np.int64)
            centers_arr = np.asarray(centers)
            for t in range(t_len):
                if len(centers_arr):
                    best = int(np.argmin(np.abs(centers_arr - t)))
                    frame_assignments[t] = assignments[best]
            frame_acc, correct, clip_total = compute_segment_frameacc_counts(data, frame_assignments)
            clips.append({
                "sequence_id": sequence_id,
                "frame_acc": float(frame_acc),
                "correct": int(correct),
                "total": int(clip_total),
                "T": int(t_len),
            })
            total_correct += correct
            total += clip_total

    return {
        "method": "frame_acc",
        "mode": "session_segments",
        "segment_root": str(segment_root),
        "custom_imu_root": "" if custom_imu_path is None else str(custom_imu_path),
        "custom_imu_split_mode": str(custom_imu_split_mode),
        "custom_imu_raw_swap_sessions": sorted(raw_swap_sessions),
        "sessions": sessions,
        "window_size": int(window_size),
        "stride": int(stride),
        "per_window_features": bool(per_window_features),
        "cosine_weight": float(cosine_weight),
        "pair_logit_weight": float(pair_logit_weight),
        "clips": clips,
        "correct": int(total_correct),
        "total": int(total),
        "frame_acc": float(np.mean([c["frame_acc"] for c in clips])) if clips else 0.0,
        "weighted_frame_acc": float(total_correct / total) if total else 0.0,
    }


def _encode_hybrid_precomputed(model, raw: torch.Tensor, vec: torch.Tensor, imu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    video_encoder = model.video_encoder
    imu_encoder = model.imu_encoder
    raw = (raw - video_encoder.raw_mu) / video_encoder.raw_sd.clamp_min(1e-6)
    vec = (vec - video_encoder.vec_mu) / video_encoder.vec_sd.clamp_min(1e-6)
    imu = (imu - imu_encoder.imu_mu) / imu_encoder.imu_sd.clamp_min(1e-6)
    video = video_encoder.fuse(torch.cat([video_encoder.raw(raw), video_encoder.vec(vec)], dim=1))
    video = F.normalize(video, dim=1)
    imu_emb = F.normalize(imu_encoder.raw(imu), dim=1)
    return imu_emb, video


class GroupTestEvaluator:
    """Sampled group matching over sequence chunks."""

    def __init__(
        self,
        group_sizes: List[int],
        num_trials: int = 50,
        chunk_windows: int = 30,
        min_chunk_windows: int = 15,
        seed: int = 42,
        shuffle_match: bool = True,
        per_subject_split: bool = False,
    ) -> None:
        self.group_sizes = group_sizes
        self.num_trials = num_trials
        self.chunk_windows = chunk_windows
        self.min_chunk_windows = min_chunk_windows
        self.seed = seed
        self.shuffle_match = shuffle_match
        self.per_subject_split = per_subject_split

    def _build_units(self, bundle: EmbeddingBundle) -> List[Dict[str, object]]:
        seq_map: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for idx, row in enumerate(bundle.rows):
            seq_name = f"{row.get('subject', '')}_{row.get('session', '')}"
            if seq_name not in seq_map:
                seq_map[seq_name] = {"imu": [], "video": []}
            seq_map[seq_name]["imu"].append(bundle.imu[idx])
            seq_map[seq_name]["video"].append(bundle.video[idx])

        units: List[Dict[str, object]] = []
        for seq_name, seq_data in sorted(seq_map.items()):
            imu_emb = np.stack(seq_data["imu"], axis=0)
            video_emb = np.stack(seq_data["video"], axis=0)
            n = min(len(imu_emb), len(video_emb))
            if n < self.min_chunk_windows:
                continue
            start = 0
            cid = 0
            while start < n:
                end = min(start + self.chunk_windows, n)
                if end - start >= self.min_chunk_windows:
                    units.append({
                        "unit_id": f"{seq_name}_c{cid:03d}",
                        "seq_name": seq_name,
                        "subject": seq_name.split("_")[0],
                        "imu_emb": imu_emb[start:end],
                        "video_emb": video_emb[start:end],
                    })
                    cid += 1
                start += self.chunk_windows
        return units

    def _eval_group(self, units: List[Dict[str, object]], group_size: int, seed: int) -> Dict[str, object]:
        if len(units) < group_size:
            return {
                "group_size": int(group_size),
                "num_units": int(len(units)),
                "num_trials": 0,
                "mean_acc": None,
                "std_acc": None,
                "mean_diag_sim": None,
                "mean_offdiag_sim": None,
                "note": f"insufficient units ({len(units)} < {group_size})",
            }

        rng = np.random.default_rng(seed)
        trial_acc: List[float] = []
        trial_diag: List[float] = []
        trial_offdiag: List[float] = []

        for _ in range(self.num_trials):
            idx = rng.choice(len(units), size=group_size, replace=False)
            selected = [units[i] for i in idx]
            if self.shuffle_match and group_size > 1:
                perm = rng.permutation(group_size)
                imu_selected = [selected[perm[i]] for i in range(group_size)]
            else:
                perm = np.arange(group_size)
                imu_selected = selected

            sim = np.zeros((group_size, group_size), dtype=np.float32)
            for i in range(group_size):
                for j in range(group_size):
                    sim[i, j] = pair_similarity(
                        imu_selected[i]["imu_emb"],
                        selected[j]["video_emb"],
                    )
            row_ind, col_ind = linear_sum_assignment(-sim)
            correct = np.sum(perm[row_ind] == col_ind) if self.shuffle_match else np.sum(row_ind == col_ind)
            trial_acc.append(float(correct) / float(group_size))
            trial_diag.append(float(np.mean(np.diag(sim))))
            if group_size > 1:
                trial_offdiag.append(float(np.mean(sim[~np.eye(group_size, dtype=bool)])))

        return {
            "group_size": int(group_size),
            "num_units": int(len(units)),
            "num_trials": int(self.num_trials),
            "mean_acc": float(np.mean(trial_acc)),
            "std_acc": float(np.std(trial_acc)),
            "mean_diag_sim": float(np.mean(trial_diag)),
            "mean_offdiag_sim": float(np.mean(trial_offdiag)) if trial_offdiag else None,
        }

    def evaluate(self, bundle: EmbeddingBundle) -> Dict[str, object]:
        units = self._build_units(bundle)
        if self.per_subject_split:
            by_subject: Dict[str, List[Dict[str, object]]] = {}
            for unit in units:
                by_subject.setdefault(str(unit["subject"]), []).append(unit)
            sampled_units: List[Dict[str, object]] = []
            rng = np.random.default_rng(self.seed)
            for subject in sorted(by_subject):
                subject_units = by_subject[subject]
                if subject_units:
                    sampled_units.append(subject_units[int(rng.integers(0, len(subject_units)))])
            eval_units = sampled_units
        else:
            eval_units = units

        results = [
            self._eval_group(eval_units, group_size, self.seed + group_size)
            for group_size in self.group_sizes
        ]
        return {
            "method": "group_test",
            "num_units": int(len(units)),
            "num_eval_units": int(len(eval_units)),
            "chunk_windows": int(self.chunk_windows),
            "min_chunk_windows": int(self.min_chunk_windows),
            "num_trials": int(self.num_trials),
            "shuffle_match": bool(self.shuffle_match),
            "per_subject_split": bool(self.per_subject_split),
            "results": results,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified FrameAcc and Group Test evaluation")
    parser.add_argument("--config", type=str, required=True, help="Workflow YAML.")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    cfg = load_cfg(cli_args.config)
    if cli_args.device:
        cfg.defrost()
        cfg.TEST.DEVICE = cli_args.device
        cfg.freeze()

    checkpoint = _resolve_checkpoint(cfg, cli_args.checkpoint)
    test_run_dir = _resolve_path(cfg.TEST.OUTPUT.OUTPUT_ROOT or repo_root() / "test") / str(cfg.TEST.OUTPUT.RUN_NAME)
    save_json = str(cli_args.save_json or test_run_dir / "results.json")
    device_name = str(cfg.TEST.DEVICE or cfg.TRAIN.DEVICE)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    frame_cfg = cfg.TEST.METRICS.FRAME_ACC
    group_cfg = cfg.TEST.METRICS.GROUP_TEST
    use_segment_frameacc = bool(str(frame_cfg.SEGMENT_ROOT).strip())
    needs_window_bundle = (bool(frame_cfg.ENABLED) and not use_segment_frameacc) or bool(group_cfg.ENABLED)
    bundle = compute_embeddings(cfg, checkpoint, device) if needs_window_bundle else None

    output: Dict[str, object] = {
        "checkpoint": str(checkpoint),
        "test_csv": str(cfg.PATHS.TEST_CSV),
        "num_rows": int(len(bundle.rows)) if bundle is not None else 0,
        "evaluations": {},
    }
    if bool(frame_cfg.ENABLED):
        if use_segment_frameacc:
            output["evaluations"]["frame_acc"] = evaluate_segment_frameacc(cfg, checkpoint, device)
        else:
            assert bundle is not None
            output["evaluations"]["frame_acc"] = FrameAccEvaluator(
                shuffle_match=bool(frame_cfg.SHUFFLE_MATCH),
                seed=int(frame_cfg.SEED),
            ).evaluate(bundle)
    if bool(group_cfg.ENABLED):
        assert bundle is not None
        output["evaluations"]["group_test"] = GroupTestEvaluator(
            group_sizes=parse_group_sizes(group_cfg.GROUP_SIZES),
            num_trials=int(group_cfg.NUM_TRIALS),
            chunk_windows=int(group_cfg.CHUNK_WINDOWS),
            min_chunk_windows=int(group_cfg.MIN_CHUNK_WINDOWS),
            seed=int(group_cfg.SEED),
            shuffle_match=bool(group_cfg.SHUFFLE_MATCH),
            per_subject_split=bool(group_cfg.PER_SUBJECT_SPLIT),
        ).evaluate(bundle)

    print(json.dumps(output, indent=2))
    if save_json:
        out = Path(save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
