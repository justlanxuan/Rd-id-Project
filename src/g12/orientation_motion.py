"""Raw G11 windows with an explicit extractor-derived turning stream."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.features.orientation import derive_2d_torso_proxy, derive_3d_torso_heading
from src.g11.raw_multiscale import RawMotionDataset, _interp_columns, _slice_time

ORIENTATION_DIM = 5
ORIENTATION_SCHEMA = "periodic_sin,periodic_cos,rate_scaled,orientation_valid,turning_activity"


class OrientationMotionDataset(RawMotionDataset):
    """Extend ``RawMotionDataset`` without changing its skeleton/IMU contract.

    Orientation is derived from the source skeleton before bbox normalization,
    then resampled to the same target grid.  This keeps the baseline input
    unchanged while giving the model a separate, auditable turning stream.
    """

    def __init__(
        self,
        specs: Sequence[dict[str, Any]],
        *,
        orientation_mode: str = "proxy",
        orientation_profile: str = "full",
        orientation_rate_scale: float = 5.0,
        **kwargs: Any,
    ) -> None:
        if orientation_mode not in {"proxy", "3d_heading", "none"}:
            raise ValueError(f"Unsupported orientation_mode={orientation_mode!r}")
        if orientation_profile not in {"full", "rate"}:
            raise ValueError(f"Unsupported orientation_profile={orientation_profile!r}")
        self.orientation_mode = orientation_mode
        self.orientation_profile = orientation_profile
        self.orientation_rate_scale = float(orientation_rate_scale)
        if self.orientation_rate_scale <= 0.0:
            raise ValueError("orientation_rate_scale must be positive")
        super().__init__(specs, **kwargs)
        self._orientation_cache: dict[int, np.ndarray] = {}

    def _source_orientation(self, row: dict[str, Any], data: dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
        key = "gt_skeleton" if "gt_skeleton" in data else "skeleton"
        values = _slice_time(np.asarray(data[key]), start, end)
        person = int(row.get("person_idx", 0))
        if values.ndim == 4:
            values = values[:, person]
        if values.ndim != 3 or values.shape[1] != 17 or values.shape[2] < 2:
            raise ValueError(f"Expected orientation skeleton [time,17,>=2], got {values.shape}")
        points = np.asarray(values[..., :3] if self.orientation_mode == "3d_heading" else values[..., :2], dtype=np.float64)
        valid = np.isfinite(points).all(axis=-1)
        visibility_key = "skeleton_visibility" if "skeleton_visibility" in data else "gt_visibility"
        if visibility_key in data:
            source_visibility = _slice_time(np.asarray(data[visibility_key]), start, end)
            if source_visibility.ndim >= 2 and source_visibility.shape[1] != 17:
                source_visibility = source_visibility[:, person]
            if source_visibility.ndim == 1:
                source_visibility = np.repeat(source_visibility[:, None], 17, axis=1)
            if source_visibility.shape != valid.shape:
                raise ValueError(f"Orientation visibility shape mismatch {source_visibility.shape} vs {valid.shape}")
            valid &= source_visibility.astype(bool)
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        timestamps = np.arange(len(points), dtype=np.float64) / float(row["_fps_hz"])
        if self.orientation_mode == "3d_heading":
            track = derive_3d_torso_heading(
                points,
                timestamps,
                visibility=valid,
                up_axis=1,
                coordinate_frame="motionbert_3d_heading_up_y",
                orientation_source="g12_motionbert_3d_torso_heading",
            )
        else:
            track = derive_2d_torso_proxy(
                points,
                timestamps,
                visibility=valid,
                coordinate_frame="extractor_image_xy",
                orientation_source="g12_extractor_2d_torso_proxy",
            )
        orientation_valid = track.orientation_valid.astype(np.float32)
        rate_valid = track.rate_valid.astype(np.float32)
        rate_scaled = np.clip(track.angle_rate / self.orientation_rate_scale, -1.0, 1.0).astype(np.float32)
        activity = np.abs(rate_scaled) * rate_valid
        values = np.stack(
            [track.angle_sin_cos[:, 0], track.angle_sin_cos[:, 1], rate_scaled, orientation_valid, activity], axis=-1
        ).astype(np.float32)
        if self.orientation_profile == "rate":
            # Absolute heading is coordinate-frame dependent across source domains.
            # Keep only signed rate, validity and activity while preserving the
            # five-channel model contract for checkpoint compatibility.
            values[:, :2] = 0.0
        values[orientation_valid < 0.5, :3] = 0.0
        values = _interp_columns(values, self.target_len)
        values[:, 3:] = (values[:, 3:] >= 0.5).astype(np.float32)
        return values.astype(np.float32)

    def _features(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if index in self._orientation_cache:
            orientation = self._orientation_cache[index]
            skeleton, imu, metadata = super()._features(index)
            return skeleton, imu, orientation, metadata
        skeleton, imu, metadata = super()._features(index)
        if self.orientation_mode == "none":
            orientation = np.zeros((self.target_len, ORIENTATION_DIM), dtype=np.float32)
        else:
            row = self.rows[index]
            path = (Path(row["_root"]) / row["npz_path"]).resolve()
            data = self._load(path)
            orientation = self._source_orientation(row, data, int(row["window_start"]), int(row["window_end"]))
        if orientation.shape != (self.target_len, ORIENTATION_DIM) or not np.isfinite(orientation).all():
            raise RuntimeError(f"Orientation feature contract failed: {orientation.shape}")
        metadata = {
            **metadata,
            "orientation_mode": self.orientation_mode,
            "orientation_profile": self.orientation_profile,
            "orientation_schema": ORIENTATION_SCHEMA,
            "orientation_rate_scale": self.orientation_rate_scale,
        }
        self._orientation_cache[index] = orientation
        return skeleton, imu, orientation, metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        skeleton, imu, orientation, metadata = self._features(index)
        row = self.rows[index]
        return {
            "skeleton": torch.from_numpy(skeleton),
            "imu": torch.from_numpy(imu),
            "orientation": torch.from_numpy(orientation),
            "index": int(index),
            "domain": str(row["_dataset"]),
            "group_key": str(row["_group_key"]),
            "identity": str(row["_identity"]),
            "subject": str(row.get("subject") or row["_identity"]),
            "session": str(row.get("session") or row.get("source_sequence") or ""),
            "split": str(row.get("split") or ""),
            "metadata": metadata,
        }
