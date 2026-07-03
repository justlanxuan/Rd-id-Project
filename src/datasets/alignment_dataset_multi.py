"""Multi-source IMU / skeleton alignment dataset.

Each sample contains two IMU windows (from two different IMU sources,
e.g. MoBind-like and realistic synthetic) paired with the same skeleton.
Used for contrastive learning where both IMUs are positives for the same
video/skeleton instance.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.alignment_dataset import (
    WindowAlignmentDataset,
    lowpass_filter_fft,
)


class WindowAlignmentDatasetMultiIMU(Dataset):
    """Window-level dataset with two paired IMU sources per skeleton."""

    def __init__(
        self,
        csv_paths: List[str | Path],
        root_dirs: List[str | Path],
        imu_stats: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        imu_sensor: Optional[str] = "R_LowArm",
        repeat_single_sensor: int = 4,
        imu_lowpass_cutoff_hz: Optional[float] = None,
        imu_lowpass_fs_hz: float = 30.0,
        return_root_trajectory: bool = False,
        root_source: str = "auto",
    ) -> None:
        if len(csv_paths) != 2 or len(root_dirs) != 2 or len(imu_stats) != 2:
            raise ValueError("This dataset expects exactly two IMU sources.")

        self.csv_paths = [Path(p) for p in csv_paths]
        self.root_dirs = [Path(r) if r is not None else p.parent for r, p in zip(root_dirs, csv_paths)]
        self.imu_stats = [
            (m.astype(np.float32) if m is not None else None,
             s.astype(np.float32) if s is not None else None)
            for m, s in imu_stats
        ]
        self.imu_sensor = imu_sensor.strip() if imu_sensor else None
        self.repeat_single_sensor = int(repeat_single_sensor)
        self.imu_lowpass_cutoff_hz = float(imu_lowpass_cutoff_hz) if imu_lowpass_cutoff_hz is not None else None
        self.imu_lowpass_fs_hz = float(imu_lowpass_fs_hz)
        self.return_root_trajectory = return_root_trajectory
        self.root_source = root_source

        rows_a = self._read_rows(self.csv_paths[0])
        rows_b = self._read_rows(self.csv_paths[1])
        self.paired_rows = self._pair_rows(rows_a, rows_b)
        self._cache: Dict[Path, Dict[str, np.ndarray]] = {}

    @staticmethod
    def _read_rows(path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            raise ValueError(f"No rows found in {path}")
        return rows

    @staticmethod
    def _pair_rows(rows_a: List[Dict[str, str]], rows_b: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
        key_cols = ["subject", "session", "split", "window_start", "window_end", "person_idx", "imu_idx"]
        b_map = {}
        for rb in rows_b:
            k = tuple(rb[c] for c in key_cols)
            if k in b_map:
                raise ValueError(f"Duplicate key in source B CSV: {k}")
            b_map[k] = rb

        pairs = []
        for ra in rows_a:
            k = tuple(ra[c] for c in key_cols)
            rb = b_map.get(k)
            if rb is None:
                raise ValueError(f"Could not find matching row in source B for key {k}")
            pairs.append((ra, rb))
        return pairs

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        if path not in self._cache:
            data = np.load(path, allow_pickle=True)
            self._cache[path] = {k: data[k] for k in data.files}
        return self._cache[path]

    def _load_imu(
        self,
        row: Dict[str, str],
        root_dir: Path,
        imu_mean: Optional[np.ndarray],
        imu_std: Optional[np.ndarray],
    ) -> np.ndarray:
        npz_rel = row["npz_path"]
        npz_path = (root_dir / npz_rel).resolve()
        data = self._load_npz(npz_path)

        st = int(row["window_start"])
        ed = int(row["window_end"])
        imu_idx = int(row.get("imu_idx", 0))

        imu = data["imu"]
        if imu.ndim == 3:
            imu = imu[st:ed, imu_idx]
        else:
            imu = imu[st:ed]

        if self.imu_sensor is not None:
            imu = WindowAlignmentDataset._single_sensor_to_48d(
                imu, self.imu_sensor, self.repeat_single_sensor
            )

        if self.imu_lowpass_cutoff_hz is not None:
            imu = lowpass_filter_fft(imu, self.imu_lowpass_cutoff_hz, self.imu_lowpass_fs_hz)

        if imu_mean is not None and imu_std is not None:
            imu = (imu - imu_mean) / np.maximum(imu_std, 1e-6)

        return imu

    def _load_skeleton(self, row: Dict[str, str], root_dir: Path) -> np.ndarray:
        npz_rel = row["npz_path"]
        npz_path = (root_dir / npz_rel).resolve()
        data = self._load_npz(npz_path)

        st = int(row["window_start"])
        ed = int(row["window_end"])
        person_idx = int(row.get("person_idx", 0))
        skeleton_source = row.get("skeleton_source", "gt")

        if skeleton_source == "gt":
            if "gt_skeleton" in data:
                skel = data["gt_skeleton"][st:ed, person_idx]
            elif "skeleton" in data:
                skel = data["skeleton"][st:ed]
            else:
                raise KeyError(f"Neither 'gt_skeleton' nor 'skeleton' found in {npz_path}")
        elif skeleton_source == "extract":
            pred_indices = data["gt_to_extract_map"][st:ed, person_idx]
            skel = np.zeros((ed - st, 17, 3), dtype=np.float32)
            extract_skeleton = data["extract_skeleton"]
            for i, pidx in enumerate(pred_indices):
                if pidx != -1:
                    skel[i] = extract_skeleton[st + i, pidx]
        else:
            raise ValueError(f"Unknown skeleton_source: {skeleton_source}")

        return skel

    def __len__(self) -> int:
        return len(self.paired_rows)

    def __getitem__(self, index: int):
        row_a, row_b = self.paired_rows[index]

        imu_a = self._load_imu(row_a, self.root_dirs[0], *self.imu_stats[0])
        imu_b = self._load_imu(row_b, self.root_dirs[1], *self.imu_stats[1])
        skel = self._load_skeleton(row_a, self.root_dirs[0])

        if imu_a.shape[0] != skel.shape[0] or imu_b.shape[0] != skel.shape[0]:
            raise ValueError(
                f"Window length mismatch at index {index}: "
                f"imu_a={imu_a.shape}, imu_b={imu_b.shape}, skel={skel.shape}"
            )

        result = {
            "imu_a": torch.from_numpy(imu_a),
            "imu_b": torch.from_numpy(imu_b),
            "skeleton": torch.from_numpy(skel),
            "subject": row_a.get("subject", ""),
            "session": row_a.get("session", ""),
            "split": row_a.get("split", ""),
        }

        if self.return_root_trajectory:
            npz_path = (self.root_dirs[0] / row_a["npz_path"]).resolve()
            data = self._load_npz(npz_path)
            root_traj = WindowAlignmentDataset._extract_root_trajectory(
                data,
                int(row_a["window_start"]),
                int(row_a["window_end"]),
                row_a.get("skeleton_source", "gt"),
                int(row_a.get("person_idx", 0)),
                self.root_source,
            )
            if root_traj is not None:
                result["root_trajectory"] = root_traj

        return result
