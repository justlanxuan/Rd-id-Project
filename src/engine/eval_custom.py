"""Evaluate IMU-video alignment on custom 2-person dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset, lowpass_filter_fft
from src.engine.common import build_alignment_model, build_loss_fn
from src.modules.matchers import IMUVideoMatcher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate custom 2-person alignment")
    p.add_argument("--test_csv", type=str, required=True, help="Path to test windows CSV")
    p.add_argument("--data_root", type=str, default=None, help="Root for relative paths in CSV")
    p.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    p.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    p.add_argument("--motionbert_ckpt", type=str, default="")
    p.add_argument("--skip_motionbert_ckpt", action="store_true")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to alignment checkpoint")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--eval_mode",
        type=str,
        default="chunk_hungarian_2person",
        choices=["chunk_hungarian_2person", "same_time_2person", "global_top1"],
    )
    p.add_argument("--chunk_windows", type=int, default=30, help="Chunk size for chunk_hungarian_2person")
    p.add_argument("--imu_stats_json", type=str, default="")
    p.add_argument("--imu_sensor", type=str, default="R_LowArm")
    p.add_argument("--repeat_single_sensor", type=int, default=4)
    p.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None, help="FFT low-pass cutoff for IMU windows in Hz; set <= 0 to disable.")
    p.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0, help="Sampling rate used by the IMU low-pass filter.")

    # Global motion options (must match training config)
    p.add_argument("--use_global_motion", action="store_true")
    p.add_argument("--global_motion_input_dim", type=int, default=2)
    p.add_argument("--global_motion_hidden_dim", type=int, default=64)
    p.add_argument("--global_motion_num_layers", type=int, default=2)
    p.add_argument("--global_motion_dropout", type=float, default=0.1)
    p.add_argument("--global_motion_input_type", type=str, default="diff_raw")
    p.add_argument("--global_motion_fusion_type", type=str, default="concat")
    p.add_argument("--global_motion_fusion_proj", action="store_true")
    p.add_argument("--global_motion_root_source", type=str, default="auto")

    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_root_trajectory_from_npz(
    arr: Dict[str, np.ndarray],
    st: int,
    ed: int,
    row: Dict[str, str],
    root_source: str,
) -> np.ndarray | None:
    person_idx = int(row.get("person_idx", row.get("imu_idx", 0)) or 0)

    if root_source in ("root_3d", "auto") and "gt_skeleton_meters" in arr:
        skel = arr["gt_skeleton_meters"][st:ed]
        if skel.ndim == 4:
            person_idx = min(person_idx, skel.shape[1] - 1)
            skel = skel[:, person_idx]
        root = skel[:, 0, :]
        return root.astype(np.float32)

    if root_source in ("bbox_center", "auto"):
        bbox_key = "extract_bboxes" if "extract_bboxes" in arr else ("gt_bboxes" if "gt_bboxes" in arr else None)
        if bbox_key is not None:
            bboxes = arr[bbox_key][st:ed]
            if bboxes.ndim == 3:
                person_idx = min(person_idx, bboxes.shape[1] - 1)
                bboxes = bboxes[:, person_idx]
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
            return np.stack([cx, cy], axis=-1).astype(np.float32)

    return None


def evaluate_global_top1(model, dataset, device, batch_size, num_workers) -> Dict[str, float]:
    """Standard window-level top-1 accuracy."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loss_fn = build_loss_fn(temperature=0.1, device=device)

    model.eval()
    total_loss = 0.0
    total_correct_ab = 0
    total_correct_ba = 0
    total_windows = 0

    with torch.no_grad():
        for batch in loader:
            imu = batch["imu"].to(device)
            skel = batch["skeleton"].to(device)
            out = model(imu=imu, skeleton=skel)

            loss = loss_fn(out["imu"], out["video"])
            total_loss += float(loss.item()) * imu.shape[0]

            z_imu = F.normalize(out["imu"], dim=-1)
            z_vid = F.normalize(out["video"], dim=-1)
            sims = torch.matmul(z_imu, z_vid.t())
            labels = torch.arange(sims.shape[0], device=sims.device)

            total_correct_ab += int((sims.argmax(dim=1) == labels).sum().item())
            total_correct_ba += int((sims.argmax(dim=0) == labels).sum().item())
            total_windows += int(sims.shape[0])

    top1_ab = float(total_correct_ab) / max(total_windows, 1)
    top1_ba = float(total_correct_ba) / max(total_windows, 1)

    return {
        "loss": float(total_loss / max(total_windows, 1)),
        "top1_ab": top1_ab,
        "top1_ba": top1_ba,
        "top1_sym": 0.5 * (top1_ab + top1_ba),
        "num_windows": total_windows,
    }


def evaluate_same_time_2person(
    model: IMUVideoMatcher,
    rows: List[Dict[str, str]],
    root_dir: Path,
    device: torch.device,
    use_global_motion: bool,
    root_source: str,
) -> Dict[str, float]:
    """Per-window 2-person pairing accuracy."""
    groups: Dict[tuple[str, int, int], List[Dict[str, str]]] = {}
    for r in rows:
        key = (str(r["session"]), int(r["window_start"]), int(r["window_end"]))
        groups.setdefault(key, []).append(r)

    pair_total = 0
    pair_correct = 0
    imu_total = 0
    imu_correct = 0
    skipped = 0

    model.eval()
    with torch.no_grad():
        for key in sorted(groups.keys()):
            g = groups[key]
            if len(g) != 2:
                skipped += 1
                continue

            g = sorted(g, key=lambda x: (str(x.get("imu_label", "")), str(x.get("person_label", ""))))

            imu_embs = []
            vid_embs = []
            for r in g:
                npz_path = (root_dir / r["npz_path"]).resolve()
                arr = np.load(npz_path)

                st = int(r["window_start"])
                ed = int(r["window_end"])
                imu = arr["imu"][st:ed].astype(np.float32)
                skel = arr["skeleton"][st:ed].astype(np.float32)
                root_traj = None
                if use_global_motion:
                    root_traj = _extract_root_trajectory_from_npz(arr, st, ed, r, root_source)

                imu_t = torch.from_numpy(imu).unsqueeze(0).to(device)
                skel_t = torch.from_numpy(skel).unsqueeze(0).to(device)
                forward_kwargs = {"imu": imu_t, "skeleton": skel_t}
                if root_traj is not None:
                    forward_kwargs["root_trajectory"] = torch.from_numpy(root_traj).unsqueeze(0).to(device)
                out = model(**forward_kwargs)

                imu_embs.append(F.normalize(out["imu"], dim=-1).squeeze(0))
                vid_embs.append(F.normalize(out["video"], dim=-1).squeeze(0))

            z_imu = torch.stack(imu_embs, dim=0)  # [2, D]
            z_vid = torch.stack(vid_embs, dim=0)  # [2, D]
            sims = torch.matmul(z_imu, z_vid.t())

            pred_cols = sims.argmax(dim=1)
            row_ok = pred_cols == torch.tensor([0, 1], device=sims.device)

            imu_correct += int(row_ok.sum().item())
            imu_total += 2
            pair_correct += int(bool(torch.all(row_ok).item()))
            pair_total += 1

    return {
        "num_pairs": int(pair_total),
        "num_skipped": int(skipped),
        "pair_top1": float(pair_correct / max(pair_total, 1)),
        "imu_top1": float(imu_correct / max(imu_total, 1)),
    }


def evaluate_chunk_hungarian_2person(
    model: IMUVideoMatcher,
    rows: List[Dict[str, str]],
    root_dir: Path,
    device: torch.device,
    chunk_windows: int,
    use_global_motion: bool,
    root_source: str,
) -> Dict[str, float]:
    """Chunk-level Hungarian matching for 2-person scenario."""
    if chunk_windows <= 0:
        raise ValueError(f"chunk_windows must be > 0, got {chunk_windows}")

    grouped: Dict[tuple[str, int, int], List[Dict[str, str]]] = {}
    for r in rows:
        key = (str(r["session"]), int(r["window_start"]), int(r["window_end"]))
        grouped.setdefault(key, []).append(r)

    by_session: Dict[str, List[tuple[int, int, List[Dict[str, str]]]]] = {}
    skipped_pair_mismatch = 0
    for (session, st, ed), g in grouped.items():
        if len(g) != 2:
            skipped_pair_mismatch += 1
            continue
        g = sorted(g, key=lambda x: (str(x.get("imu_label", "")), str(x.get("person_label", ""))))
        by_session.setdefault(session, []).append((st, ed, g))

    # Encode all embeddings
    session_cache: Dict[str, List[tuple[np.ndarray, np.ndarray]]] = {}
    model.eval()
    with torch.no_grad():
        for session, items in by_session.items():
            items = sorted(items, key=lambda x: (x[0], x[1]))
            seq: List[tuple[np.ndarray, np.ndarray]] = []
            for _st, _ed, g in items:
                imu_embs = []
                vid_embs = []
                for r in g:
                    npz_path = (root_dir / r["npz_path"]).resolve()
                    arr = np.load(npz_path)

                    st = int(r["window_start"])
                    ed = int(r["window_end"])
                    imu = arr["imu"][st:ed].astype(np.float32)
                    skel = arr["skeleton"][st:ed].astype(np.float32)
                    root_traj = None
                    if use_global_motion:
                        root_traj = _extract_root_trajectory_from_npz(arr, st, ed, r, root_source)

                    imu_t = torch.from_numpy(imu).unsqueeze(0).to(device)
                    skel_t = torch.from_numpy(skel).unsqueeze(0).to(device)
                    forward_kwargs = {"imu": imu_t, "skeleton": skel_t}
                    if root_traj is not None:
                        forward_kwargs["root_trajectory"] = torch.from_numpy(root_traj).unsqueeze(0).to(device)
                    out = model(**forward_kwargs)

                    imu_embs.append(F.normalize(out["imu"], dim=-1).squeeze(0).cpu().numpy())
                    vid_embs.append(F.normalize(out["video"], dim=-1).squeeze(0).cpu().numpy())

                seq.append((np.stack(imu_embs, axis=0), np.stack(vid_embs, axis=0)))
            session_cache[session] = seq

    chunk_total = 0
    chunk_correct = 0
    imu_total = 0
    imu_correct = 0
    skipped_short = 0

    for session, seq in session_cache.items():
        n = len(seq)
        if n < chunk_windows:
            skipped_short += 1
            continue

        for s in range(0, n - chunk_windows + 1):
            e = s + chunk_windows
            chunk = seq[s:e]

            sim = np.zeros((2, 2), dtype=np.float32)
            for i in range(2):
                for j in range(2):
                    vals = [float(np.dot(imu2[i], vid2[j])) for imu2, vid2 in chunk]
                    sim[i, j] = float(np.mean(vals))

            row_ind, col_ind = linear_sum_assignment(-sim)
            assign = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

            c0 = int(assign.get(0, -1) == 0)
            c1 = int(assign.get(1, -1) == 1)
            imu_correct += c0 + c1
            imu_total += 2
            chunk_correct += int(c0 == 1 and c1 == 1)
            chunk_total += 1

    return {
        "num_chunks": int(chunk_total),
        "num_skipped": int(skipped_pair_mismatch + skipped_short),
        "chunk_top1": float(chunk_correct / max(chunk_total, 1)),
        "imu_top1": float(imu_correct / max(imu_total, 1)),
        "chunk_windows": int(chunk_windows),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    imu_mean = None
    imu_std = None
    if args.imu_stats_json:
        stats = json.loads(Path(args.imu_stats_json).read_text())
        imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(stats["imu_std"], dtype=np.float32)

    imu_sensor = args.imu_sensor.strip() if args.imu_sensor else None
    ds = WindowAlignmentDataset(
        args.test_csv,
        root_dir=args.data_root,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
        imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
        imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
    )

    model, _ = build_alignment_model(args, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    rows = _read_csv_rows(Path(args.test_csv))
    root_dir = Path(args.data_root) if args.data_root else Path(args.test_csv).parent

    if args.eval_mode == "global_top1":
        metrics = evaluate_global_top1(model, ds, device, args.batch_size, args.num_workers)
    elif args.eval_mode == "same_time_2person":
        metrics = evaluate_same_time_2person(
            model,
            rows,
            root_dir,
            device,
            use_global_motion=bool(args.use_global_motion),
            root_source=str(args.global_motion_root_source),
        )
    elif args.eval_mode == "chunk_hungarian_2person":
        metrics = evaluate_chunk_hungarian_2person(
            model,
            rows,
            root_dir,
            device,
            args.chunk_windows,
            use_global_motion=bool(args.use_global_motion),
            root_source=str(args.global_motion_root_source),
        )
    else:
        raise ValueError(f"Unknown eval_mode: {args.eval_mode}")

    print(json.dumps(metrics, indent=2))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
