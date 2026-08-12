from __future__ import annotations

import csv

import numpy as np

from preprocess.adapters import DATASET_ADAPTERS, DatasetAdapter, build_dataset_adapter
from preprocess.datasets import custom as preprocess_custom
from src.datasets import WindowAlignmentDataset


def test_raw_dataset_adapters_are_configurable_classes(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("preprocess:\n  dataset: custom\n", encoding="utf-8")

    assert DATASET_ADAPTERS.names() == ("custom", "egohumans", "totalcapture")
    for name in DATASET_ADAPTERS.names():
        assert isinstance(build_dataset_adapter(name, config_path), DatasetAdapter)


def test_preprocess_package_exports_dataset_helpers():
    assert hasattr(preprocess_custom, "load_custom_split_7d_sequence")


def test_window_alignment_dataset_reads_standard_npz(tmp_path):
    npz_path = tmp_path / "seq.npz"
    np.savez(
        npz_path,
        imu=np.arange(10 * 7, dtype=np.float32).reshape(10, 7),
        skeleton=np.zeros((10, 17, 3), dtype=np.float32),
    )

    csv_path = tmp_path / "windows.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "session",
                "split",
                "npz_path",
                "window_start",
                "window_end",
                "skeleton_source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "S1",
                "session": "sess",
                "split": "train",
                "npz_path": "seq.npz",
                "window_start": "2",
                "window_end": "6",
                "skeleton_source": "gt",
            }
        )

    dataset = WindowAlignmentDataset(csv_path, root_dir=tmp_path, imu_sensor=None)
    sample = dataset[0]

    assert sample["imu"].shape == (4, 7)
    assert sample["skeleton"].shape == (4, 17, 3)
    assert sample["subject"] == "S1"
    assert sample["group_key"] == "sess|2|6"
