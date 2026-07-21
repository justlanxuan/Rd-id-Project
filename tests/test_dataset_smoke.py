from __future__ import annotations

import csv

import numpy as np

from src.datasets import WindowAlignmentDataset
from src.datasets.alignment_dataset import WindowAlignmentDataset as LegacyWindowAlignmentDataset


def test_alignment_dataset_import_compatibility():
    assert WindowAlignmentDataset is LegacyWindowAlignmentDataset


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
