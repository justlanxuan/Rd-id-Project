from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from preprocess.datasets.custom import run_preprocess as run_custom_preprocess
from preprocess.derived import DerivedDataSpec
from preprocess.derived.registry import apply_transform
from src.config import get_cfg_defaults
from src.datasets import WindowAlignmentDataset
from src.datasets.custom_session import load_custom_tracklet_session
from src.engine.evaluate import load_segment_eval_inputs
from src.features.imu import (
    CANONICAL_7D_CHANNELS,
    feature_spec_from_cfg,
    feature_spec_from_config,
    select_imu_features,
)
from src.models.checkpoint import load_model_checkpoint, model_checkpoint_metadata
from src.models.registry import build_model


def _model_config(*, channels: tuple[str, ...], mode: str = "raw"):
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.TRAIN.MODEL.HYBRID_HIDDEN = 16
    cfg.TRAIN.MODEL.HYBRID_TOKEN_HEADS = 4
    cfg.TRAIN.MODEL.HYBRID_TEMPORAL_LAYERS = 1
    cfg.TRAIN.MODEL.HYBRID_DROPOUT = 0.0
    cfg.TRAIN.MODEL.HYBRID_IMU_FEATURE_MODE = mode
    cfg.TRAIN.IMU_FEATURES = channels
    cfg.freeze()
    return cfg


def test_feature_spec_preserves_explicit_order_and_mapping_compatibility() -> None:
    channels = ("quat_w", "quat_x", "quat_y", "quat_z", "acc_x", "acc_y", "acc_z")
    spec = feature_spec_from_config(channels=channels)
    mapping_spec = feature_spec_from_cfg({"train": {"imu_features": list(channels)}})

    assert spec.channels == channels
    assert mapping_spec.sha256 == spec.sha256
    assert spec.input_dim == 7

    values = np.arange(14, dtype=np.float32).reshape(2, 7)
    selected = select_imu_features(values, CANONICAL_7D_CHANNELS, spec)
    assert np.array_equal(selected, values[:, [3, 4, 5, 6, 0, 1, 2]])


def test_window_dataset_selects_named_channels_in_configured_order(tmp_path) -> None:
    channels = (
        "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z"
    )
    values = np.arange(2 * 5 * 10, dtype=np.float32).reshape(5, 2, 10)
    np.savez(
        tmp_path / "sequence.npz",
        imu=values,
        imu_channels=np.asarray(channels, dtype=object),
        gt_skeleton=np.zeros((5, 2, 17, 3), dtype=np.float32),
    )
    with (tmp_path / "windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["npz_path", "window_start", "window_end", "imu_idx", "person_idx", "skeleton_source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "npz_path": "sequence.npz",
                "window_start": "1",
                "window_end": "4",
                "imu_idx": "1",
                "person_idx": "1",
                "skeleton_source": "gt",
            }
        )

    spec = feature_spec_from_config(channels=("gyro_z", "acc_x", "quat_w"))
    dataset = WindowAlignmentDataset(
        tmp_path / "windows.csv",
        root_dir=tmp_path,
        imu_sensor=None,
        imu_feature_spec=spec,
    )
    sample = dataset[0]

    assert sample["imu"].shape == (3, 3)
    assert torch.equal(sample["imu"], torch.from_numpy(values[1:4, 1][:, [5, 0, 6]]))


def test_legacy_48d_is_an_explicit_compatibility_view() -> None:
    values = np.zeros((2, 48), dtype=np.float32)
    values[:, 2 * 9 : 3 * 9] = np.eye(3, dtype=np.float32).reshape(1, 9)
    values[:, 36 + 2 * 3 : 36 + 3 * 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    spec = feature_spec_from_config(view="canonical_7d")

    selected = select_imu_features(values, tuple(f"legacy_{i}" for i in range(48)), spec)

    assert np.allclose(selected[:, :3], [1.0, 2.0, 3.0])
    assert np.allclose(selected[:, 3:], [1.0, 0.0, 0.0, 0.0])


def test_segment_and_full_session_loaders_keep_named_feature_width(tmp_path) -> None:
    channels = (
        "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z"
    )
    imu = np.zeros((4, 1, 10), dtype=np.float32)
    imu[..., 0] = 1.0
    imu[..., 6] = 1.0
    skeleton = np.zeros((4, 1, 17, 3), dtype=np.float32)
    segment = tmp_path / "custom_demo_seg0.npz"
    np.savez(
        segment,
        sequence_id=np.asarray("custom_demo_seg0", dtype=object),
        frame_ids=np.arange(4, dtype=np.int64),
        imu=imu,
        imu_channels=np.asarray(channels, dtype=object),
        imu_ids=np.asarray([0], dtype=np.int64),
        extract_person_ids=np.asarray([0], dtype=np.int64),
        extract_skeleton=skeleton,
        extract_visibility=np.ones((4, 1), dtype=bool),
    )
    spec = feature_spec_from_config(channels=("gyro_x", "acc_x", "quat_w"))

    _data, _sequence_id, _pose, selected = load_segment_eval_inputs(
        segment,
        None,
        imu_feature_spec=spec,
    )

    aligned = tmp_path / "custom_demo.npz"
    np.savez(
        aligned,
        sequence_id=np.asarray("custom_demo", dtype=object),
        frame_ids=np.arange(4, dtype=np.int64),
        imu=imu,
        imu_channels=np.asarray(channels, dtype=object),
        imu_ids=np.asarray([0], dtype=np.int64),
        gt_person_ids=np.asarray([0], dtype=np.int64),
        gt_bboxes=np.zeros((4, 1, 4), dtype=np.float32),
        gt_visibility=np.ones((4, 1), dtype=bool),
    )
    tracklets = tmp_path / "tracklets.json"
    tracklets.write_text("[]")
    session = load_custom_tracklet_session(aligned, tracklets, imu_feature_spec=spec)

    assert selected.shape == (4, 1, 3)
    assert session.imu.shape == (4, 1, 3)
    assert session.imu_channels == spec.channels


def test_named_derived_transform_preserves_non_acc_channels() -> None:
    channels = np.asarray(
        ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"],
        dtype=object,
    )
    imu = np.zeros((8, 1, 10), dtype=np.float32)
    imu[..., 0:3] = 4.0
    imu[..., 3] = 1.0
    imu[..., 6] = 1.0
    payload = {"imu": imu, "imu_channels": channels}
    spec = DerivedDataSpec(enabled=True, transforms=("imu_acc_noise",), imu_acc_noise_std=0.2)

    output = apply_transform("imu_acc_noise", payload, np.random.default_rng(3), spec)

    assert np.array_equal(output["imu"][..., :3], imu[..., :3])
    assert np.array_equal(output["imu"][..., 6:], imu[..., 6:])
    assert not np.array_equal(output["imu"][..., 3:6], imu[..., 3:6])


def test_custom_adapter_can_write_named_acc_gyro_quat_channels(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    csv_path = raw_root / "demo_imu.csv"
    headers = [
        "epoch_ms", "四元数0()", "四元数1()", "四元数2()", "四元数3()",
        "加速度X(g)", "加速度Y(g)", "加速度Z(g)",
        "角速度X(deg/s)", "角速度Y(deg/s)", "角速度Z(deg/s)",
    ]
    rows = [
        [index * 10, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 90.0]
        for index in range(4)
    ]
    csv_path.write_text(
        ",".join(headers) + "\n" + "\n".join(",".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
preprocess:
  dataset: custom
  raw_root: {raw_root}
  output: {tmp_path / 'prepared'}
  imu:
    use_48d: true
    output_view: acc_gyro_quat
""",
        encoding="utf-8",
    )

    output_root = run_custom_preprocess(config)

    with np.load(output_root / "sequences/custom_demo_imu.npz", allow_pickle=True) as data:
        assert data["imu"].shape == (4, 1, 10)
        assert tuple(str(value) for value in data["imu_channels"]) == (
            "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z"
        )


def test_hybrid_model_width_and_checkpoint_contract_follow_feature_spec() -> None:
    channels = ("gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z")
    cfg = _model_config(channels=channels, mode="dynamic")
    model, name = build_model(cfg, torch.device("cpu"))
    output = model(torch.randn(2, 8, 10), torch.randn(2, 8, 17, 3))
    metadata = model_checkpoint_metadata(name, model)

    assert model.imu_encoder.feature_dim == 30
    assert output["imu"].shape == (2, 16)
    assert metadata["imu_feature_spec"]["channels"] == list(channels)
    assert metadata["imu_feature_spec_sha256"] == model.imu_feature_spec.sha256

    legacy_cfg = _model_config(channels=CANONICAL_7D_CHANNELS)
    legacy_model, legacy_name = build_model(legacy_cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="feature contract"):
        load_model_checkpoint(
            legacy_model,
            legacy_name,
            {**metadata, "model": model.state_dict()},
        )
