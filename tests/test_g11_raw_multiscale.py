from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.g10.global_encoder import GlobalMotionMatcher
from src.g11.raw_multiscale import RAW_IMU_DIM, RAW_SKELETON_DIM, RawMotionDataset
from tools.g11.build_duration_manifest import build_duration_rows
from tools.g11.build_raw_freeze_record import build_record


def _write_csv(path: Path, row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _source_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    root = tmp_path / "source"
    sidecar_root = tmp_path / "sidecar"
    (root / "sequences").mkdir(parents=True)
    sidecar_root.mkdir()
    time, people = 12, 2
    skeleton = np.arange(time * people * 17 * 3, dtype=np.float32).reshape(time, people, 17, 3) / 100.0
    imu = np.zeros((time, people, 7), dtype=np.float32)
    np.savez(
        root / "sequences" / "sample.npz",
        sequence_id=np.asarray("sample", dtype=object),
        gt_skeleton=skeleton,
        gt_visibility=np.ones((time, people), dtype=bool),
        imu=imu,
    )
    acc = np.stack([np.arange(time), np.arange(time) * 2, np.arange(time) * 3], axis=-1).astype(np.float32)
    acc = np.stack([acc, acc + 1], axis=1)
    gyro = acc * 0.01
    np.savez(
        sidecar_root / "sample.npz",
        acceleration_mps2=acc,
        gyroscope_rads=gyro,
        provenance=np.asarray("synthetic", dtype=object),
        sensor_location=np.asarray("left_wrist", dtype=object),
    )
    csv_path = root / "windows.csv"
    _write_csv(
        csv_path,
        {
            "npz_path": "sequences/sample.npz",
            "window_start": "2",
            "window_end": "10",
            "person_idx": "1",
            "imu_idx": "1",
            "session": "source_session",
            "source_sequence": "sample",
            "source_window_start": "2",
            "source_window_end": "10",
        },
    )
    spec: dict[str, object] = {
        "dataset": "source",
        "csv": str(csv_path),
        "root": str(root),
        "fps_hz": 10.0,
        "gyro_sidecar_root": str(sidecar_root),
    }
    return spec, sidecar_root


def test_raw_dataset_maps_full_sequence_to_xy_visibility_and_acc_gyro(tmp_path: Path) -> None:
    spec, _ = _source_fixture(tmp_path)
    dataset = RawMotionDataset([spec], target_len=8, skeleton_normalize="bbox", imu_normalize="separate_zscore")
    item = dataset[0]
    assert item["skeleton"].shape == (8, RAW_SKELETON_DIM)
    assert item["imu"].shape == (8, RAW_IMU_DIM)
    assert torch.isfinite(item["skeleton"]).all()
    assert torch.isfinite(item["imu"]).all()
    assert item["metadata"]["visibility_rate"] == pytest.approx(1.0)
    assert item["metadata"]["channels"] == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    assert torch.allclose(item["imu"][:, :3].mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(item["imu"][:, 3:].mean(dim=0), torch.zeros(3), atol=1e-5)


def test_raw_dataset_supports_folded_custom_window_and_identity(tmp_path: Path) -> None:
    root = tmp_path / "custom"
    sidecar_root = tmp_path / "sidecar"
    (root / "sequences").mkdir(parents=True)
    sidecar_root.mkdir()
    skeleton = np.random.default_rng(0).normal(size=(8, 17, 2)).astype(np.float32)
    np.savez(root / "sequences" / "custom_p1_0_8.npz", skeleton=skeleton, imu=np.zeros((8, 7), dtype=np.float32))
    np.savez(
        sidecar_root / "custom_p1_0_8.npz",
        acceleration_mps2=np.ones((8, 1, 3), dtype=np.float32),
        gyroscope_rads=np.full((8, 1, 3), 2.0, dtype=np.float32),
        provenance=np.asarray("measured", dtype=object),
        sensor_location=np.asarray("left_wrist", dtype=object),
    )
    csv_path = root / "windows.csv"
    _write_csv(
        csv_path,
        {
            "npz_path": "sequences/custom_p1_0_8.npz",
            "window_start": "0",
            "window_end": "8",
            "person_idx": "0",
            "imu_idx": "0",
            "source_person": "1",
            "session": "23",
            "source_sequence": "custom_sequence",
            "source_window_start": "0",
            "source_window_end": "8",
        },
    )
    dataset = RawMotionDataset(
        [{"dataset": "custom23", "csv": csv_path, "root": root, "fps_hz": 30, "gyro_sidecar_root": sidecar_root, "session_filter": "23"}],
        target_len=8,
        imu_normalize="none",
    )
    item = dataset[0]
    assert item["identity"] == "1"
    assert item["metadata"]["provenance"] == "measured"
    assert torch.allclose(item["imu"][:, :3], torch.ones(8, 3))
    assert torch.allclose(item["imu"][:, 3:], torch.full((8, 3), 2.0))


def test_raw_dataset_requires_explicit_gyro_sidecar(tmp_path: Path) -> None:
    spec, sidecar_root = _source_fixture(tmp_path)
    (sidecar_root / "sample.npz").unlink()
    dataset = RawMotionDataset([spec], target_len=8)
    with pytest.raises(FileNotFoundError, match="Missing required gyro sidecar"):
        dataset[0]


def test_raw_dataset_audits_explicit_missing_sidecar_skip(tmp_path: Path) -> None:
    spec, sidecar_root = _source_fixture(tmp_path)
    (sidecar_root / "sample.npz").unlink()
    # Keep one valid row so the dataset can report the skipped row rather than
    # failing because the whole requested evaluation split is empty.
    valid = sidecar_root / "valid.npz"
    np.savez(
        valid,
        acceleration_mps2=np.ones((8, 1, 3), dtype=np.float32),
        gyroscope_rads=np.ones((8, 1, 3), dtype=np.float32),
    )
    csv_path = Path(spec["csv"])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid_row = dict(rows[0])
    valid_row["npz_path"] = "sequences/valid.npz"
    valid_row["source_sequence"] = "valid"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([rows[0], valid_row])
    np.savez(
        Path(spec["root"]) / "sequences" / "valid.npz",
        sequence_id=np.asarray("valid", dtype=object),
        gt_skeleton=np.ones((8, 1, 17, 3), dtype=np.float32),
        gt_visibility=np.ones((8, 1), dtype=bool),
    )
    spec["skip_missing_sidecar"] = "true"
    dataset = RawMotionDataset([spec], target_len=8)
    assert len(dataset) == 1
    assert dataset.sidecar_audit == [
        {
            "dataset": "source",
            "input_rows": 2,
            "kept_rows": 1,
            "skipped_rows": 1,
            "skipped_sidecars": [str(sidecar_root / "sample.npz")],
        }
    ]


def test_raw_dataset_rejects_native_frame_duration_mismatch(tmp_path: Path) -> None:
    spec, _ = _source_fixture(tmp_path)
    with pytest.raises(ValueError, match="duration mismatch"):
        RawMotionDataset([spec], target_len=8, window_seconds=0.5)


def test_raw_multiscale_matcher_accepts_51d_skeleton_and_6d_imu() -> None:
    model = GlobalMotionMatcher(
        skeleton_dim=RAW_SKELETON_DIM,
        imu_dim=RAW_IMU_DIM,
        hidden=16,
        embedding_dim=8,
        temporal_mode="multiscale",
        window_seconds=0.8,
    )
    output = model(torch.randn(3, 24, RAW_SKELETON_DIM), torch.randn(3, 24, RAW_IMU_DIM))
    assert output["skeleton"].shape == (3, 8)
    assert output["imu"].shape == (3, 8)
    assert torch.allclose(output["skeleton_scale_weights"].sum(dim=-1), torch.ones(3), atol=1e-6)


def test_duration_manifest_uses_real_seconds_and_full_sequence_paths(tmp_path: Path) -> None:
    spec, _ = _source_fixture(tmp_path)
    with Path(spec["csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output, manifest = build_duration_rows(
        rows,
        root=Path(spec["root"]),
        fps_hz=10.0,
        window_seconds=0.8,
        stride_seconds=0.4,
    )
    assert [(row["window_start"], row["window_end"]) for row in output] == [("0", "8"), ("4", "12")]
    assert manifest["window_frames"] == 8
    assert manifest["stride_frames"] == 4
    assert all(row["npz_path"] == "sequences/sample.npz" for row in output)


def test_raw_freeze_uses_three_seed_custom23_mean(tmp_path: Path) -> None:
    runs: list[Path] = []
    for fusion, values in {"mean": (0.50, 0.51, 0.52), "hierarchical_attention": (0.52, 0.53, 0.54)}.items():
        for seed, value in enumerate(values):
            run = tmp_path / f"{fusion}_{seed}"
            run.mkdir()
            (run / "best.pt").write_bytes(f"{fusion}-{seed}".encode())
            metric = {
                "schema_version": "g11.raw_multiscale_run.v1",
                "config": {"seed": seed, "temporal_mode": "multiscale", "multiscale_fusion": fusion},
                "history": [
                    {
                        "epoch": 1,
                        "eval": {
                            "per_domain": {
                                "custom23": {"frame_acc": value, "correct": round(value * 100), "total": 100},
                                "egohumans_realistic": {"frame_acc": 0.4, "correct": 40, "total": 100},
                            }
                        },
                    }
                ],
            }
            (run / "metrics.json").write_text(json.dumps(metric), encoding="utf-8")
            runs.append(run)
    record = build_record(runs)
    assert record["primary_fusion"] == "hierarchical_attention"
    assert record["candidate_ranking"][0]["custom23_mean"] == pytest.approx(0.53)
    assert record["status"] == "frozen_before_custom57_22_24"
