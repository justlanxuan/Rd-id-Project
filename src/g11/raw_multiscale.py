"""Auditable raw skeleton/IMU6 dataset for G11 E3."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

N_JOINTS = 17
RAW_SKELETON_DIM = N_JOINTS * 3  # xy + visibility per joint
RAW_IMU_DIM = 6  # acceleration xyz + gyroscope xyz


def _interp_columns(values: np.ndarray, target_len: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or len(array) < 1:
        raise ValueError(f"Expected non-empty [time, channels], got {array.shape}")
    if int(target_len) < 2:
        raise ValueError(f"target_len must be >=2, got {target_len!r}")
    if len(array) == int(target_len):
        return array.copy()
    if len(array) == 1:
        return np.repeat(array, int(target_len), axis=0)
    source = np.linspace(0.0, 1.0, len(array))
    target = np.linspace(0.0, 1.0, int(target_len))
    return np.stack([np.interp(target, source, array[:, column]) for column in range(array.shape[1])], axis=-1).astype(np.float32)


def _window_key(row: dict[str, str]) -> str:
    explicit = str(row.get("candidate_group_id", "")).strip()
    if explicit:
        return f"candidate|{explicit}"
    sequence = row.get("source_sequence") or row.get("sequence_id") or row.get("npz_path", "")
    start = row.get("source_window_start") or row.get("window_start", "")
    end = row.get("source_window_end") or row.get("window_end", "")
    return "|".join((str(sequence), str(start), str(end)))


def _identity(row: dict[str, str]) -> str:
    if str(row.get("candidate_group_id", "")).strip():
        return str(row.get("candidate_index") or row.get("source_sequence") or row.get("npz_path") or "0")
    return str(row.get("source_person") or row.get("imu_idx") or row.get("person_idx") or "0")


def _slice_time(values: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0 or end <= start:
        raise ValueError(f"Invalid window [{start}, {end})")
    array = np.asarray(values)
    if end <= len(array):
        return array[start:end]
    expected = end - start
    if len(array) == expected:
        return array
    raise ValueError(f"Window [{start}, {end}) is incompatible with array length {len(array)}")


def _select_person(values: np.ndarray, person: int, *, expected_last_dim: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if expected_last_dim is not None and array.shape[-1] != expected_last_dim:
        raise ValueError(f"Expected last dimension {expected_last_dim}, got {array.shape}")
    if array.ndim >= 3 and array.shape[1] != N_JOINTS:
        if person < 0 or person >= array.shape[1]:
            raise ValueError(f"person index {person} out of range for {array.shape}")
        return array[:, person]
    return array


def _bbox_normalize(xy: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    output = np.zeros_like(xy, dtype=np.float32)
    for frame in range(len(xy)):
        valid = visibility[frame]
        if not valid.any():
            continue
        visible = xy[frame, valid]
        low = visible.min(axis=0)
        high = visible.max(axis=0)
        center = 0.5 * (low + high)
        scale = max(float(np.max(high - low)), 1e-5)
        output[frame, valid] = (visible - center) / scale
    return output


def _separate_imu_zscore(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    for start, end in ((0, 3), (3, 6)):
        part = output[:, start:end]
        output[:, start:end] = (part - part.mean(axis=0, keepdims=True)) / np.maximum(part.std(axis=0, keepdims=True), 1e-5)
    return output


class RawMotionDataset(Dataset):
    """Load canonical/folded windows as 51D skeleton and 6D IMU sequences."""

    def __init__(
        self,
        specs: Sequence[dict[str, Any]],
        *,
        target_len: int = 24,
        skeleton_normalize: str = "bbox",
        imu_normalize: str = "separate_zscore",
        window_seconds: float | None = None,
    ) -> None:
        if not specs:
            raise ValueError("RawMotionDataset requires at least one spec")
        if skeleton_normalize not in {"none", "bbox"}:
            raise ValueError(f"Unsupported skeleton_normalize={skeleton_normalize!r}")
        if imu_normalize not in {"none", "separate_zscore"}:
            raise ValueError(f"Unsupported imu_normalize={imu_normalize!r}")
        self.target_len = int(target_len)
        if self.target_len < 2:
            raise ValueError("target_len must be >=2")
        self.skeleton_normalize = skeleton_normalize
        self.imu_normalize = imu_normalize
        self.window_seconds = None if window_seconds is None else float(window_seconds)
        if self.window_seconds is not None and not 0.0 < self.window_seconds <= 10.0:
            raise ValueError(f"window_seconds must be in (0, 10], got {window_seconds!r}")
        self.rows: list[dict[str, Any]] = []
        self.sidecar_audit: list[dict[str, Any]] = []
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self._feature_cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for spec in specs:
            required = {"dataset", "csv", "root", "fps_hz", "gyro_sidecar_root"}
            missing = required - set(spec)
            if missing:
                raise ValueError(f"Raw spec missing fields {sorted(missing)}")
            csv_path = Path(spec["csv"]).expanduser().resolve()
            root = Path(spec["root"]).expanduser().resolve()
            sidecar = Path(spec["gyro_sidecar_root"]).expanduser().resolve()
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if spec.get("session_filter"):
                rows = [row for row in rows if row.get("session") == str(spec["session_filter"])]
            input_rows = len(rows)
            skipped_sidecars: list[str] = []
            if spec.get("skip_missing_sidecar"):
                kept: list[dict[str, str]] = []
                for row in rows:
                    candidate = sidecar / Path(row["npz_path"]).name
                    if candidate.is_file():
                        kept.append(row)
                    else:
                        skipped_sidecars.append(str(candidate))
                rows = kept
            if not rows:
                raise ValueError(f"No rows found for raw spec {csv_path}")
            self.sidecar_audit.append(
                {
                    "dataset": str(spec["dataset"]),
                    "input_rows": input_rows,
                    "kept_rows": len(rows),
                    "skipped_rows": len(skipped_sidecars),
                    "skipped_sidecars": skipped_sidecars,
                }
            )
            for row in rows:
                if self.window_seconds is not None:
                    native_frames = int(row["window_end"]) - int(row["window_start"])
                    native_seconds = native_frames / float(spec["fps_hz"])
                    tolerance = 1.0 / float(spec["fps_hz"]) + 1e-6
                    if abs(native_seconds - self.window_seconds) > tolerance:
                        raise ValueError(
                            f"Raw window duration mismatch for {row.get('npz_path')}: "
                            f"{native_frames} frames / {float(spec['fps_hz']):g} Hz = {native_seconds:g}s, "
                            f"expected {self.window_seconds:g}s"
                        )
                copied: dict[str, Any] = dict(row)
                copied["_dataset"] = str(spec["dataset"])
                copied["_root"] = str(root)
                copied["_sidecar"] = str(sidecar)
                copied["_fps_hz"] = float(spec["fps_hz"])
                copied["_group_key"] = _window_key(row)
                copied["_identity"] = _identity(row)
                self.rows.append(copied)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({str(row["_dataset"]) for row in self.rows}))

    def group_indices(self) -> dict[tuple[str, str], list[int]]:
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            groups[(str(row["_dataset"]), str(row["_group_key"]))].append(index)
        return dict(groups)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._cache:
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as data:
                self._cache[path] = {key: data[key] for key in data.files}
        return self._cache[path]

    @staticmethod
    def _sequence_id(data: dict[str, np.ndarray], path: Path) -> str:
        if "sequence_id" in data:
            return str(np.asarray(data["sequence_id"]).item())
        return path.stem

    def _skeleton(self, row: dict[str, Any], data: dict[str, np.ndarray], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        key = "gt_skeleton" if "gt_skeleton" in data else "skeleton"
        if key not in data:
            raise KeyError("Raw dataset requires gt_skeleton or skeleton")
        values = _slice_time(np.asarray(data[key]), start, end)
        person = int(row.get("person_idx", 0))
        if values.ndim == 4:
            values = values[:, person]
        if values.ndim != 3 or values.shape[1] != N_JOINTS or values.shape[2] < 2:
            raise ValueError(f"Expected skeleton [time,17,>=2], got {values.shape}")
        xy = np.asarray(values[..., :2], dtype=np.float32)
        visibility = np.isfinite(xy).all(axis=-1)
        visibility_key = "skeleton_visibility" if "skeleton_visibility" in data else "gt_visibility"
        if visibility_key in data:
            source_visibility = _slice_time(np.asarray(data[visibility_key]), start, end)
            if source_visibility.ndim >= 2 and source_visibility.shape[1] != N_JOINTS:
                source_visibility = source_visibility[:, person]
            if source_visibility.ndim == 1:
                source_visibility = np.repeat(source_visibility[:, None], N_JOINTS, axis=1)
            if source_visibility.shape != visibility.shape:
                raise ValueError(f"Visibility shape mismatch {source_visibility.shape} vs {visibility.shape}")
            visibility &= source_visibility.astype(bool)
        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        xy[~visibility] = 0.0
        if not visibility.any():
            raise ValueError("Raw skeleton window has no visible joints")
        if self.skeleton_normalize == "bbox":
            xy = _bbox_normalize(xy, visibility)
        xy = _interp_columns(xy.reshape(len(xy), -1), self.target_len).reshape(self.target_len, N_JOINTS, 2)
        visibility_values = _interp_columns(visibility.astype(np.float32), self.target_len) >= 0.5
        xy[~visibility_values] = 0.0
        features = np.concatenate([xy, visibility_values[..., None].astype(np.float32)], axis=-1).reshape(self.target_len, -1)
        return features.astype(np.float32), visibility_values

    def _sidecar_path(self, row: dict[str, Any], data: dict[str, np.ndarray], path: Path) -> Path:
        root = Path(row["_sidecar"])
        direct = root / path.name
        if direct.is_file():
            return direct
        sequence = self._sequence_id(data, path)
        candidate = root / f"{sequence}.npz"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Missing required gyro sidecar for {path}: tried {direct} and {candidate}")

    def _imu(self, row: dict[str, Any], data: dict[str, np.ndarray], path: Path, start: int, end: int) -> tuple[np.ndarray, dict[str, Any]]:
        sidecar_path = self._sidecar_path(row, data, path)
        side = self._load(sidecar_path)
        if "acceleration_mps2" not in side or "gyroscope_rads" not in side:
            raise KeyError(f"Sidecar lacks acceleration/gyroscope: {sidecar_path}")
        acc = _slice_time(np.asarray(side["acceleration_mps2"]), start, end)
        gyro = _slice_time(np.asarray(side["gyroscope_rads"]), start, end)
        imu_index = int(row.get("imu_idx", 0))
        if acc.ndim == 3:
            selected = imu_index if acc.shape[1] > 1 else 0
            acc = acc[:, selected]
            gyro = gyro[:, selected]
        if acc.ndim != 2 or gyro.ndim != 2 or acc.shape[1] != 3 or gyro.shape[1] != 3 or len(acc) != len(gyro):
            raise ValueError(f"Invalid IMU sidecar shapes acc={acc.shape}, gyro={gyro.shape}")
        values = np.concatenate([acc, gyro], axis=-1).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite raw IMU values in {sidecar_path}")
        if self.imu_normalize == "separate_zscore":
            values = _separate_imu_zscore(values)
        values = _interp_columns(values, self.target_len)
        provenance = str(np.asarray(side.get("provenance", "unknown")).item())
        location = str(np.asarray(side.get("sensor_location", "unknown")).item())
        return values.astype(np.float32), {
            "sidecar": str(sidecar_path),
            "provenance": provenance,
            "sensor_location": location,
            "channels": ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        }

    def _features(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if index in self._feature_cache:
            return self._feature_cache[index]
        row = self.rows[index]
        path = (Path(row["_root"]) / row["npz_path"]).resolve()
        data = self._load(path)
        start, end = int(row["window_start"]), int(row["window_end"])
        skeleton, visibility = self._skeleton(row, data, start, end)
        imu, provenance = self._imu(row, data, path, start, end)
        if skeleton.shape != (self.target_len, RAW_SKELETON_DIM) or imu.shape != (self.target_len, RAW_IMU_DIM):
            raise RuntimeError(f"Raw feature shape contract failed: skeleton={skeleton.shape}, imu={imu.shape}")
        metadata = {
            **provenance,
            "source_path": str(path),
            "fps_hz": float(row["_fps_hz"]),
            "visibility_rate": float(visibility.mean()),
            "skeleton_normalize": self.skeleton_normalize,
            "imu_normalize": self.imu_normalize,
        }
        self._feature_cache[index] = (skeleton, imu, metadata)
        return self._feature_cache[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        skeleton, imu, metadata = self._features(index)
        row = self.rows[index]
        return {
            "skeleton": torch.from_numpy(skeleton),
            "imu": torch.from_numpy(imu),
            "index": int(index),
            "domain": str(row["_dataset"]),
            "group_key": str(row["_group_key"]),
            "identity": str(row["_identity"]),
            "metadata": metadata,
        }
