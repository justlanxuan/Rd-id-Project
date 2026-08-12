"""Build canonical Custom cyclic-inner-validation window folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from preprocess.datasets.custom import load_custom_rawcsv_7d_sequence
from tools.g6.matrix import CUSTOM_FOLDS


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "subject",
        "session",
        "split",
        "npz_path",
        "window_start",
        "window_end",
        "window_len",
        "skeleton_source",
        "person_idx",
        "imu_idx",
        "source_sequence",
        "source_person",
        "source_window_start",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _session(sequence_id: str) -> str:
    return sequence_id.split("_seg", 1)[0].split("custom_", 1)[1]


def build_custom_folds(
    segment_root: str | Path,
    output_root: str | Path,
    raw_imu_root: str | Path,
    *,
    window_len: int = 24,
    stride: int = 24,
    raw_swap_sessions: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Write four immutable fold directories without mutating an existing root."""
    segments = Path(segment_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    raw_root = Path(raw_imu_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite Custom fold root: {destination}")
    destination.mkdir(parents=True)

    sequence_payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for path in sorted(segments.glob("custom_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            sequence_id = str(data["sequence_id"].item())
            session = _session(sequence_id)
            pose = np.asarray(data["extract_skeleton"][:, :, :, :2], dtype=np.float32)
            visibility = np.asarray(data["extract_visibility"], dtype=bool)
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            n_people = int(pose.shape[1])
            mapping = str(data["imu_person_map"].item()) if "imu_person_map" in data.files else None
            imu = load_custom_rawcsv_7d_sequence(
                raw_root,
                session,
                frame_ids,
                imu_person_map=mapping,
                n_persons=n_people,
            )
        if raw_swap_sessions and session in raw_swap_sessions:
            imu = imu[:, ::-1].copy()
        sequence_payloads[sequence_id] = (pose, visibility, imu)

    summary: dict[str, dict[str, int]] = {}
    for fold_id, fold in sorted(CUSTOM_FOLDS.items()):
        test_session = str(fold["test_session"])
        val_session = str(fold["val_session"])
        train_sessions = {str(value) for value in fold["train_sessions"]}
        fold_dir = destination / f"fold{fold_id}_{test_session}"
        sequence_dir = fold_dir / "sequences"
        sequence_dir.mkdir(parents=True)
        rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

        for sequence_id, (pose, visibility, imu) in sorted(sequence_payloads.items()):
            session = _session(sequence_id)
            split = "test" if session == test_session else "val" if session == val_session else "train"
            if split == "train" and session not in train_sessions:
                raise ValueError(f"Session {session} is outside fold {fold_id}")
            for person in range(int(pose.shape[1])):
                for start in range(0, int(pose.shape[0]) - window_len + 1, stride):
                    end = start + window_len
                    if not visibility[start:end, person].any():
                        continue
                    relative = f"sequences/{sequence_id}_p{person}_{start}_{end}.npz"
                    np.savez_compressed(
                        fold_dir / relative,
                        skeleton=pose[start:end, person],
                        imu=imu[start:end, person],
                    )
                    rows_by_split[split].append(
                        {
                            "subject": f"P{person}",
                            "session": session,
                            "split": split,
                            "npz_path": relative,
                            "window_start": 0,
                            "window_end": window_len,
                            "window_len": window_len,
                            "skeleton_source": "gt",
                            "person_idx": 0,
                            "imu_idx": 0,
                            "source_sequence": sequence_id,
                            "source_person": person,
                            "source_window_start": start,
                        }
                    )

        counts = {split: len(rows) for split, rows in rows_by_split.items()}
        for split, rows in rows_by_split.items():
            _write_csv(fold_dir / f"windows_{split}.csv", rows)
        (fold_dir / "slice_summary.json").write_text(
            json.dumps(
                {
                    "test_session": test_session,
                    "val_session": val_session,
                    "train_sessions": sorted(train_sessions),
                    "window_len": window_len,
                    "stride": stride,
                    "segment_root": str(segments),
                    "raw_imu_root": str(raw_root),
                    "raw_swap_sessions": sorted(raw_swap_sessions or set()),
                    "counts": counts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary[test_session] = counts
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--raw-imu-root", required=True)
    parser.add_argument("--window-len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--raw-swap-sessions", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    swaps = {value.strip() for value in args.raw_swap_sessions.split(",") if value.strip()}
    summary = build_custom_folds(
        args.segment_root,
        args.output_root,
        args.raw_imu_root,
        window_len=args.window_len,
        stride=args.stride,
        raw_swap_sessions=swaps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
