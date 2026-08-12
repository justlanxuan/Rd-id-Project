"""Dataset-adapter registry."""

from __future__ import annotations

from pathlib import Path

from src.core import Registry

from .base import DatasetAdapter
from .official import CustomAdapter, EgoHumansAdapter, TotalCaptureAdapter

DATASET_ADAPTERS: Registry[DatasetAdapter] = Registry("dataset adapter")
DATASET_ADAPTERS.register(
    "totalcapture",
    aliases=("totalcapture_vicon", "totalcapture_synthetic", "totalcapture_synthetic_imu"),
)(TotalCaptureAdapter)
DATASET_ADAPTERS.register("egohumans", aliases=("egohumans_full",))(EgoHumansAdapter)
DATASET_ADAPTERS.register("custom")(CustomAdapter)


def build_dataset_adapter(name: str, config_path: str | Path) -> DatasetAdapter:
    return DATASET_ADAPTERS.build(name, config_path)
