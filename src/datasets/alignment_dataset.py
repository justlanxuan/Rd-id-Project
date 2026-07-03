"""Window-level IMU/Video alignment dataset."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def lowpass_filter_fft(signal: np.ndarray, cutoff_hz: float | None, fs_hz: float) -> np.ndarray:
    """Apply a simple FFT-domain low-pass filter along the time axis.

    The input is expected to have time as axis 0, e.g. [T, D] or [T, N, D].
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        return signal.astype(np.float32, copy=False)

    if fs_hz <= 0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")

    time_len = signal.shape[0]
    if time_len <= 1:
        return signal.astype(np.float32, copy=False)

    nyquist = fs_hz / 2.0
    effective_cutoff = float(cutoff_hz)
    if effective_cutoff >= nyquist:
        clipped = max(nyquist * 0.95, 1e-6)
        warnings.warn(
            f"Requested IMU low-pass cutoff {effective_cutoff:.3f} Hz exceeds Nyquist {nyquist:.3f} Hz; "
            f"clipping to {clipped:.3f} Hz.",
            RuntimeWarning,
            stacklevel=2,
        )
        effective_cutoff = clipped

    freq = np.fft.rfftfreq(time_len, d=1.0 / fs_hz)
    spectrum = np.fft.rfft(signal, axis=0)
    spectrum[freq > effective_cutoff, ...] = 0.0
    filtered = np.fft.irfft(spectrum, n=time_len, axis=0)
    return filtered.astype(np.float32, copy=False)


class WindowAlignmentDataset(Dataset):
    """Window-level IMU/Video alignment dataset from CSV index."""

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path | None = None,
        imu_mean: Optional[np.ndarray] = None,
        imu_std: Optional[np.ndarray] = None,
        imu_sensor: Optional[str] = "R_LowArm",
        repeat_single_sensor: int = 4,
        imu_lowpass_cutoff_hz: Optional[float] = None,
        imu_lowpass_fs_hz: float = 30.0,
        return_root_trajectory: bool = False,
        root_source: Literal["auto", "bbox_center", "bbox_full", "bbox_vel", "root_3d"] = "auto",
        per_session_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.root_dir = Path(root_dir) if root_dir is not None else self.csv_path.parent
        self.rows = self._read_rows(self.csv_path)
        self._cache: Dict[Path, Dict[str, np.ndarray]] = {}
        self.imu_mean = imu_mean.astype(np.float32) if imu_mean is not None else None
        self.imu_std = imu_std.astype(np.float32) if imu_std is not None else None
        self.imu_sensor = imu_sensor.strip() if imu_sensor else None
        self.repeat_single_sensor = int(repeat_single_sensor)
        self.imu_lowpass_cutoff_hz = float(imu_lowpass_cutoff_hz) if imu_lowpass_cutoff_hz is not None else None
        self.imu_lowpass_fs_hz = float(imu_lowpass_fs_hz)
        self.return_root_trajectory = return_root_trajectory
        self.root_source = root_source
        self.per_session_stats = per_session_stats

        if self.imu_sensor is not None and self.repeat_single_sensor <= 0:
            raise ValueError(f"repeat_single_sensor must be > 0, got {self.repeat_single_sensor}")

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

    def __len__(self) -> int:
        return len(self.rows)

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        if path not in self._cache:
            data = np.load(path, allow_pickle=True)
            self._cache[path] = {k: data[k] for k in data.files}
        return self._cache[path]

    def __getitem__(self, index: int):
        row = self.rows[index]
        npz_rel = row["npz_path"]
        npz_path = (self.root_dir / npz_rel).resolve()
        data = self._load_npz(npz_path)

        st = int(row["window_start"])
        ed = int(row["window_end"])

        imu_idx = int(row.get("imu_idx", 0))
        person_idx = int(row.get("person_idx", 0))
        skeleton_source = row.get("skeleton_source", "gt")

        # IMU: format [T, N_imu, 48] (single-person [T, 48] also handled)
        imu = data["imu"]
        if imu.ndim == 3:
            imu = imu[st:ed, imu_idx]
        else:
            imu = imu[st:ed]

        # Skeleton
        if skeleton_source == "gt":
            if "gt_skeleton" in data:
                skel = data["gt_skeleton"][st:ed, person_idx]
            elif "skeleton" in data:
                # MotionBERT alignment NPZ compatibility: single skeleton stream at key "skeleton".
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

        if self.imu_sensor is not None:
            imu = self._single_sensor_to_48d(imu, self.imu_sensor, self.repeat_single_sensor)

        if self.imu_lowpass_cutoff_hz is not None:
            imu = lowpass_filter_fft(imu, self.imu_lowpass_cutoff_hz, self.imu_lowpass_fs_hz)

        # Per-session normalization takes precedence over global normalization
        session = row.get("session", "")
        if self.per_session_stats is not None and session in self.per_session_stats:
            sess_mean, sess_std = self.per_session_stats[session]
            imu = (imu - sess_mean) / np.maximum(sess_std, 1e-6)
        elif self.imu_mean is not None and self.imu_std is not None:
            imu = (imu - self.imu_mean) / np.maximum(self.imu_std, 1e-6)

        if imu.shape[0] != skel.shape[0]:
            raise ValueError(f"Window length mismatch in {npz_path}: {imu.shape} vs {skel.shape}")

        result = {
            "imu": torch.from_numpy(imu),
            "skeleton": torch.from_numpy(skel),
            "subject": row.get("subject", ""),
            "session": row.get("session", ""),
            "split": row.get("split", ""),
            "domain": row.get("domain", ""),
        }

        if self.return_root_trajectory:
            root_traj = self._extract_root_trajectory(
                data, st, ed, skeleton_source, person_idx, self.root_source
            )
            if root_traj is not None:
                result["root_trajectory"] = root_traj

        return result

    @staticmethod
    def _extract_root_trajectory(
        data: Dict[str, np.ndarray],
        st: int,
        ed: int,
        skeleton_source: str,
        person_idx: int,
        source_type: Literal["auto", "bbox_center", "bbox_full", "bbox_vel", "root_3d"],
    ) -> Optional[torch.Tensor]:
        """Extract root trajectory from NPZ data.

        Supported source_types:
            - bbox_center: [cx, cy]
            - bbox_full:   [cx, cy, w, h]  (center + size)
            - bbox_vel:    [cx, cy, vx, vy] (center + velocity)
            - root_3d:     [x, y, z] from GT skeleton
            - auto:        try root_3d first, then bbox_center
        """
        if source_type in ("root_3d", "auto"):
            key = "gt_skeleton_meters"
            if key in data:
                skel = data[key][st:ed, person_idx]
                root = skel[:, 0, :]
                return torch.from_numpy(root.astype(np.float32))

        bbox_key = "gt_bboxes" if skeleton_source == "gt" else "extract_bboxes"
        if bbox_key in data:
            bboxes = data[bbox_key][st:ed, person_idx]
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
            cx_t = torch.from_numpy(cx).float()
            cy_t = torch.from_numpy(cy).float()

            if source_type in ("bbox_center", "auto"):
                return torch.stack([cx_t, cy_t], dim=-1)

            if source_type == "bbox_full":
                w = torch.from_numpy(np.abs(bboxes[:, 2] - bboxes[:, 0])).float()
                h = torch.from_numpy(np.abs(bboxes[:, 3] - bboxes[:, 1])).float()
                return torch.stack([cx_t, cy_t, w, h], dim=-1)

            if source_type == "bbox_vel":
                vx = np.zeros_like(cx, dtype=np.float32)
                vy = np.zeros_like(cy, dtype=np.float32)
                vx[1:] = cx[1:] - cx[:-1]
                vy[1:] = cy[1:] - cy[:-1]
                return torch.stack([
                    cx_t, cy_t,
                    torch.from_numpy(vx), torch.from_numpy(vy)
                ], dim=-1)

        return None

    @staticmethod
    def _single_sensor_to_48d(imu: np.ndarray, sensor_name: str, repeat_single_sensor: int) -> np.ndarray:
        order = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]
        if sensor_name not in order:
            raise ValueError(f"Unsupported sensor_name={sensor_name}. Must be one of {order}")

        if repeat_single_sensor != 4:
            raise ValueError(
                "Alignment IMU encoder expects 48D input. "
                f"Use repeat_single_sensor=4 for single-sensor mode; got {repeat_single_sensor}."
            )

        k = order.index(sensor_name)
        rot = imu[:, k * 9 : (k + 1) * 9]
        acc = imu[:, 36 + k * 3 : 36 + (k + 1) * 3]

        out = np.zeros((imu.shape[0], 48), dtype=np.float32)
        for i in range(4):
            out[:, i * 9 : (i + 1) * 9] = rot
            out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
        return out
