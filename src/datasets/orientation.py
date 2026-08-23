"""Config-driven construction of extractor-derived orientation windows."""

from __future__ import annotations

from typing import Any

from src.data.specs import load_specs
from src.datasets.orientation_motion import OrientationMotionDataset


def orientation_specs_from_cfg(cfg: Any, split: str) -> list[dict[str, Any]]:
    orientation = cfg.TRAIN.ORIENTATION
    values = getattr(orientation, f"{str(split).upper()}_SPECS", ())
    return load_specs(values)


def build_orientation_dataset(cfg: Any, split: str) -> OrientationMotionDataset:
    orientation = cfg.TRAIN.ORIENTATION
    specs = orientation_specs_from_cfg(cfg, split)
    return OrientationMotionDataset(
        specs,
        target_len=int(orientation.TARGET_LEN),
        skeleton_normalize="bbox",
        imu_normalize="separate_zscore",
        window_seconds=float(orientation.WINDOW_SECONDS) if orientation.WINDOW_SECONDS else None,
        orientation_mode=str(orientation.MODE),
        orientation_profile=str(orientation.PROFILE),
        orientation_rate_scale=float(orientation.ORIENTATION_RATE_SCALE),
    )


__all__ = ["OrientationMotionDataset", "build_orientation_dataset", "orientation_specs_from_cfg"]
