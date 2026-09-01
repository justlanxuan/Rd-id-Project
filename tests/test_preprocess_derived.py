from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from preprocess.adapters import PreprocessArtifact
from preprocess.common.imu import parse_imu_csv_with_gyro
from preprocess.common.imu_conditioning import IMU_CONDITIONER_REGISTRY, condition_imu, run_madgwick_imu
from preprocess.datasets.custom import run_preprocess as run_custom_preprocess
from preprocess.derived import DerivedDataSpec, derive_sequences
from preprocess.derived.registry import DERIVED_TRANSFORM_REGISTRY, apply_transform
from src.config import load_cfg
from src.workflow.stages import PreprocessStage


def _payload() -> dict[str, np.ndarray]:
    skeleton = np.zeros((4, 1, 17, 3), dtype=np.float32)
    parents = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
    for joint, parent in enumerate(parents):
        if parent >= 0:
            skeleton[:, 0, joint] = skeleton[:, 0, parent] + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    imu = np.zeros((4, 1, 7), dtype=np.float32)
    imu[..., 0] = 1.0
    imu[..., 3] = 1.0
    return {
        "schema_version": np.array("1.0", dtype=object),
        "dataset": np.array("custom", dtype=object),
        "sequence_id": np.array("custom_demo", dtype=object),
        "frame_ids": np.arange(4, dtype=np.int64),
        "imu": imu,
        "imu_channels": np.asarray(
            ["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"], dtype=object
        ),
        "imu_ids": np.asarray([0], dtype=np.int64),
        "gt_person_ids": np.asarray([0], dtype=np.int64),
        "gt_visibility": np.ones((4, 1), dtype=bool),
        "gt_bboxes": np.zeros((4, 1, 4), dtype=np.float32),
        "gt_skeleton": skeleton,
        "gt_skeleton_meters": skeleton.copy(),
    }


def test_derived_registry_exposes_initial_semantic_transforms() -> None:
    assert DERIVED_TRANSFORM_REGISTRY.names() == (
        "identity",
        "imu_acc_dropout_hold",
        "imu_acc_lowpass",
        "imu_acc_noise",
        "imu_acc_spike",
        "imu_mount_rotation",
        "imu_quat_repair",
        "skeleton_bone_scale",
        "skeleton_coord_noise",
        "skeleton_joint_dropout",
        "skeleton_temporal_jitter",
        "skeleton_track_fragmentation",
    )


def test_imu_acc_noise_does_not_change_quaternion() -> None:
    payload = _payload()
    spec = DerivedDataSpec(enabled=True, transforms=("imu_acc_noise",), imu_acc_noise_std=0.25)
    first = apply_transform("imu_acc_noise", payload, np.random.default_rng(7), spec)
    second = apply_transform("imu_acc_noise", payload, np.random.default_rng(7), spec)

    assert np.array_equal(first["imu"], second["imu"])
    assert np.array_equal(first["imu"][..., 3:], payload["imu"][..., 3:])
    assert not np.array_equal(first["imu"][..., :3], payload["imu"][..., :3])


def test_imu_stress_variants_preserve_quaternion_channels() -> None:
    payload = _payload()
    payload["imu"][..., 0] = np.arange(4, dtype=np.float32)[:, None]
    spike_spec = DerivedDataSpec(enabled=True, transforms=("imu_acc_spike",), imu_acc_spike_ratio=0.5)
    dropout_spec = DerivedDataSpec(
        enabled=True,
        transforms=("imu_acc_dropout_hold",),
        imu_acc_dropout_duration=2,
        imu_acc_dropout_segments=1,
    )

    spiked = apply_transform("imu_acc_spike", payload, np.random.default_rng(7), spike_spec)
    dropped = apply_transform("imu_acc_dropout_hold", payload, np.random.default_rng(7), dropout_spec)

    assert np.array_equal(spiked["imu"][..., 3:], payload["imu"][..., 3:])
    assert np.array_equal(dropped["imu"][..., 3:], payload["imu"][..., 3:])
    assert dropped["derived_imu_validity"].shape == (4, 1)
    assert not dropped["derived_imu_validity"].all()


def test_mount_rotation_preserves_acceleration_norm_and_quaternion_geometry() -> None:
    payload = _payload()
    payload["imu"][..., 0] = 0.0
    payload["imu"][..., 1] = 1.0
    spec = DerivedDataSpec(enabled=True, transforms=("imu_mount_rotation",), imu_mount_euler_xyz_deg=(90.0, 0.0, 0.0))
    output = apply_transform("imu_mount_rotation", payload, np.random.default_rng(7), spec)

    assert np.allclose(np.linalg.norm(output["imu"][..., :3], axis=-1), 1.0)
    assert np.allclose(np.linalg.norm(output["imu"][..., 3:], axis=-1), 1.0, atol=1e-6)
    assert not np.allclose(output["imu"][..., 3:], payload["imu"][..., 3:])


def test_quaternion_repair_normalizes_and_makes_sign_continuous() -> None:
    payload = _payload()
    payload["imu"][1, ..., 3:] = -payload["imu"][1, ..., 3:]
    payload["imu"][2, ..., 3:] *= 2.0
    spec = DerivedDataSpec(enabled=True, transforms=("imu_quat_repair",))
    output = apply_transform("imu_quat_repair", payload, np.random.default_rng(7), spec)

    quaternions = output["imu"][..., 3:]
    assert np.allclose(np.linalg.norm(quaternions, axis=-1), 1.0)
    assert np.all(np.sum(quaternions[:-1] * quaternions[1:], axis=-1) >= 0.0)


def test_madgwick_imu_is_causal_and_keeps_quaternions_valid() -> None:
    timestamps = np.arange(8, dtype=np.float64) * 10.0
    initial = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (8, 1))
    acceleration = np.tile(np.asarray([0.0, 0.0, 9.80665], dtype=np.float64), (8, 1))
    gyro = np.zeros((8, 3), dtype=np.float64)
    gyro[1:, 2] = 1.0

    output = run_madgwick_imu(timestamps, initial, acceleration, gyro, beta=0.033)

    assert output.shape == (8, 4)
    assert np.isfinite(output).all()
    assert np.allclose(np.linalg.norm(output, axis=-1), 1.0, atol=1e-6)
    assert not np.allclose(output[-1], output[0])
    assert IMU_CONDITIONER_REGISTRY.names() == ("identity", "madgwick6")
    with pytest.raises(ValueError, match="requires native-rate gyro"):
        condition_imu("madgwick6", timestamps, initial, acceleration)


def test_identity_conditioner_does_not_modify_legacy_quaternion_values() -> None:
    timestamps = np.arange(3, dtype=np.float64)
    quaternions = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    output = condition_imu("identity", timestamps, quaternions, np.zeros((3, 3), dtype=np.float32))

    assert np.array_equal(output, quaternions)
    assert output is not quaternions


def test_bone_scale_is_sequence_consistent_and_keeps_root_fixed() -> None:
    payload = _payload()
    spec = DerivedDataSpec(
        enabled=True,
        transforms=("skeleton_bone_scale",),
        skeleton_bone_scale_min=2.0,
        skeleton_bone_scale_max=2.0,
    )
    output = apply_transform("skeleton_bone_scale", payload, np.random.default_rng(7), spec)

    assert np.array_equal(output["gt_skeleton"][..., 0, :], payload["gt_skeleton"][..., 0, :])
    assert np.allclose(output["gt_skeleton"][..., 1, :] - output["gt_skeleton"][..., 0, :], [2.0, 0.0, 0.0])
    assert np.allclose(
        output["gt_skeleton"][..., 16, :] - output["gt_skeleton"][..., 15, :],
        [2.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "transform_name",
    ["skeleton_coord_noise", "skeleton_joint_dropout", "skeleton_temporal_jitter", "skeleton_track_fragmentation"],
)
def test_rb_skeleton_variants_preserve_sequence_contract(transform_name: str) -> None:
    payload = _payload()
    spec = DerivedDataSpec(enabled=True, transforms=(transform_name,))
    output = apply_transform(transform_name, payload, np.random.default_rng(7), spec)

    assert output["gt_skeleton"].shape == payload["gt_skeleton"].shape
    assert np.isfinite(output["gt_skeleton"]).all()
    assert np.array_equal(output["gt_skeleton"][..., 0, :], payload["gt_skeleton"][..., 0, :])


def test_custom_raw_preprocess_can_select_rg23_conditioner(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    csv_path = raw_root / "demo_imu.csv"
    headers = [
        "epoch_ms",
        "四元数0()",
        "四元数1()",
        "四元数2()",
        "四元数3()",
        "加速度X(g)",
        "加速度Y(g)",
        "加速度Z(g)",
        "角速度X(deg/s)",
        "角速度Y(deg/s)",
        "角速度Z(deg/s)",
    ]
    rows = [
        [index * 10, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 90.0]
        for index in range(8)
    ]
    csv_path.write_text(
        ",".join(headers) + "\n" + "\n".join(",".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project: rg23_custom_test
preprocess:
  dataset: custom
  raw_root: {raw_root}
  output: {tmp_path / 'out' / 'video_manifest.csv'}
  imu:
    use_48d: false
    conditioner: madgwick6
    conditioner_beta: 0.033
""",
        encoding="utf-8",
    )

    output_root = run_custom_preprocess(config_path)
    with np.load(output_root / "sequences/custom_demo_imu.npz", allow_pickle=True) as data:
        imu = np.asarray(data["imu"])
        assert imu.shape == (8, 1, 7)
        assert np.allclose(np.linalg.norm(imu[..., 3:], axis=-1), 1.0, atol=1e-6)
    meta = json.loads((output_root / "sequences/custom_demo_imu.json").read_text(encoding="utf-8"))
    assert meta["imu_conditioner"] == "madgwick6"
    assert parse_imu_csv_with_gyro(csv_path)[3].shape == (8, 3)


def test_derive_sequences_writes_variant_provenance(tmp_path) -> None:
    source = tmp_path / "canonical"
    sequence_dir = source / "sequences"
    sequence_dir.mkdir(parents=True)
    np.savez_compressed(sequence_dir / "custom_demo.npz", **_payload())
    (sequence_dir / "custom_demo.json").write_text(json.dumps({"data_layer": "standardized"}), encoding="utf-8")
    (source / "video_manifest.csv").write_text("video_path\n", encoding="utf-8")

    output = derive_sequences(
        source,
        tmp_path / "derived" / "rc_mount",
        {
            "enabled": True,
            "name": "rc_mount",
            "transforms": ["imu_mount_rotation"],
            "imu_mount_euler_xyz_deg": [10.0, 0.0, 0.0],
        },
    )

    assert (output / "sequences/custom_demo.npz").exists()
    assert (output / "video_manifest.csv").exists()
    manifest = json.loads((output / "derived_manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((output / "sequences/custom_demo.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "rc_mount"
    assert manifest["records"][0]["transforms"] == ["imu_mount_rotation"]
    assert meta["data_layer"] == "derived"
    assert meta["derived_variant"] == "rc_mount"


def test_enabled_derived_data_routes_default_slice_root(tmp_path: Path) -> None:
    config_path = tmp_path / "derived.yaml"
    config_path.write_text(
        """
project: derived_config_test
preprocess:
  dataset: custom
  derived:
    enabled: true
    name: rc_mount
    transforms: [imu_mount_rotation]
""",
        encoding="utf-8",
    )

    cfg = load_cfg(config_path)

    assert cfg.PREPROCESS.DERIVED.OUTPUT.endswith("/custom/preprocessed/derived_config_test/derived/rc_mount")
    assert cfg.SLICE.ROOT == cfg.PREPROCESS.DERIVED.OUTPUT
    assert cfg.SLICE.OUT_DIR == cfg.PREPROCESS.DERIVED.OUTPUT
    assert cfg.PATHS.DATA_ROOT == cfg.PREPROCESS.DERIVED.OUTPUT


def test_preprocess_stage_slices_the_derived_root(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "canonical"
    sequence_dir = source / "sequences"
    sequence_dir.mkdir(parents=True)
    np.savez_compressed(sequence_dir / "custom_demo.npz", **_payload())

    config_path = tmp_path / "derived.yaml"
    config_path.write_text(
        f"""
project: derived_stage_test
preprocess:
  dataset: custom
  output: {source / 'video_manifest.csv'}
  derived:
    enabled: true
    name: rc_acc
    output: {tmp_path / 'rc_acc'}
    transforms: [imu_acc_noise]
    imu_acc_noise_std: 0.1
slice:
  window_len: 2
  stride: 1
  train_sessions: demo
  val_sessions: demo
  test_sessions: demo
""",
        encoding="utf-8",
    )

    class FakeAdapter:
        def preprocess(self, *, output_dir=None, manifest_csv=None):
            return PreprocessArtifact("custom", source, prepared=False)

    monkeypatch.setattr(
        "src.workflow.stages.build_dataset_adapter",
        lambda _dataset, _config_path: FakeAdapter(),
    )

    state = PreprocessStage(config_path).run()

    assert state["derived_data"] is True
    assert Path(state["data_root"]) == tmp_path / "rc_acc"
    assert (tmp_path / "rc_acc/windows_train.csv").exists()
