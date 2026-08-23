import numpy as np
import pytest

from preprocess.common.gyro import parse_custom_csv
from src.features.global_motion import (
    COCO17_JOINTS,
    FeatureContractError,
    derive_trajectory_features,
    extract_global_anchors,
    extract_imu_views,
    spectral_summary,
)
from tools.g10.build_feature_manifest import build_manifest
from tools.g10.run_physical_screen import _cca_similarity, _coherence, _gp_likelihood, run_screen


def _skeleton(t_len=8):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(t_len, 17, 2)).astype(np.float64)
    return values


def test_translation_is_visible_in_global_anchors_and_similarity_descriptor():
    base = np.repeat(_skeleton(1), 8, axis=0)
    shift = np.arange(len(base), dtype=float)[:, None, None] * np.array([0.2, -0.1])
    bundle = extract_global_anchors(base + shift, joint_names=COCO17_JOINTS, coordinate_space="pixel")
    np.testing.assert_allclose(bundle.trajectories["A3_shoulder_midpoint"][1] - bundle.trajectories["A3_shoulder_midpoint"][0], [0.2, -0.1])
    np.testing.assert_allclose(bundle.trajectories["A8_robust_similarity_transform"][1, :2], [0.2, -0.1], atol=1e-7)
    assert bundle.validity["A8_robust_similarity_transform"][1]


def test_similarity_descriptor_rejects_one_locally_moving_joint():
    base = np.repeat(_skeleton(1), 3, axis=0)
    translated = base.copy()
    translated[1] += np.array([0.5, 0.25])
    translated_clean = translated.copy()
    translated[1, 9] += np.array([100.0, -100.0])
    clean = extract_global_anchors(translated_clean, joint_names=COCO17_JOINTS)
    changed = extract_global_anchors(translated, joint_names=COCO17_JOINTS)
    np.testing.assert_allclose(
        clean.trajectories["A8_robust_similarity_transform"][1, :2],
        changed.trajectories["A8_robust_similarity_transform"][1, :2],
        atol=1e-5,
    )


def test_derivatives_use_seconds_not_frame_indices():
    timestamps = np.array([0.0, 0.5, 1.5, 3.0])
    trajectory = np.stack([timestamps**2, np.zeros_like(timestamps)], axis=-1)
    features, masks = derive_trajectory_features(trajectory, timestamps)
    np.testing.assert_allclose(features["velocity"][1:, 0], [0.5, 2.0, 4.5])
    assert not masks["velocity"][0]
    assert masks["acceleration"][2]


def test_sinusoid_spectral_summary_stays_in_common_band():
    fs = 30.0
    timestamps = np.arange(150) / fs
    signal = np.stack([np.sin(2 * np.pi * 2.0 * timestamps), np.zeros_like(timestamps)], axis=-1)
    summary = spectral_summary(signal, timestamps, band_hz=(0.0, 4.5))
    assert summary["valid"]
    assert abs(summary["dominant_hz"] - 2.0) < 0.25
    with pytest.raises(FeatureContractError):
        spectral_summary(signal, timestamps, band_hz=(0.0, 20.0))


def test_coherence_is_cross_spectral_not_an_alias_for_ncc():
    t = np.arange(120, dtype=float) / 30.0
    x = np.sin(2 * np.pi * 2.0 * t)
    y = np.sin(2 * np.pi * 2.0 * t + 0.7)
    z = np.random.default_rng(4).normal(size=len(t))
    assert _coherence(x, y) > _coherence(x, z)


def test_cca_similarity_handles_multichannel_imu_view():
    t = np.arange(120, dtype=float) / 30.0
    x = np.sin(2 * np.pi * 2.0 * t)
    y = np.stack([x + 0.01 * np.cos(t), np.random.default_rng(8).normal(size=len(t))], axis=-1)
    z = np.random.default_rng(9).normal(size=(len(t), 2))
    assert _cca_similarity(x, y) > _cca_similarity(x, z)


def test_gp_style_likelihood_prefers_matching_observation():
    t = np.arange(40, dtype=float) / 20.0
    x = np.sin(2 * np.pi * 2.0 * t)
    matching = x + 0.03 * np.cos(t)
    random = np.random.default_rng(10).normal(size=len(t))
    assert _gp_likelihood(x, matching) > _gp_likelihood(x, random)


def test_visibility_and_semantic_layout_fail_loudly():
    values = _skeleton(4)
    visibility = np.ones((4, 17), dtype=bool)
    visibility[2, 9] = False
    bundle = extract_global_anchors(values, joint_names=COCO17_JOINTS, visibility=visibility)
    assert not bundle.validity["A0_left_wrist"][2]
    with pytest.raises(FeatureContractError, match="left wrist"):
        extract_global_anchors(values, joint_names=[f"joint_{i}" for i in range(17)])


def test_imu_views_support_acc_only_and_named_full_contract():
    timestamps = np.arange(6, dtype=float) / 30.0
    acc = np.stack([timestamps, timestamps**2, np.ones_like(timestamps)], axis=-1)
    acc_views = extract_imu_views(
        acc,
        timestamps,
        channel_names=["acc_x", "acc_y", "acc_z"],
        sensor_location="left_wrist",
        provenance="custom_device",
    )
    assert set(acc_views) == {"I0_acc", "I1_acc_magnitude", "I2_acc_changes"}
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(timestamps), 1))
    gyro = np.zeros_like(acc)
    full = np.concatenate([acc, gyro, quat], axis=-1)
    views = extract_imu_views(
        full,
        timestamps,
        channel_names=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "quat_w", "quat_x", "quat_y", "quat_z"],
        sensor_location="LeftWrist",
        provenance="egohumans_realistic_smpl_kinematics",
    )
    assert {"I0_acc", "I3_gyro", "I5_acc_gyro", "I6_acc_quat", "I7_acc_gyro_quat"} <= set(views)
    assert views["I7_acc_gyro_quat"].values.shape == (6, 10)


def test_invalid_quaternion_is_explicitly_masked():
    timestamps = np.arange(4, dtype=float) / 30.0
    values = np.zeros((4, 7), dtype=float)
    values[:, 0] = 1.0
    values[:, 3] = 1.0
    values[2, 3:7] = 0.0
    views = extract_imu_views(
        values,
        timestamps,
        channel_names=["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"],
        sensor_location="left_wrist",
        provenance="custom_device",
    )
    assert not views["I6_acc_quat"].validity[2]
    assert np.isfinite(views["I6_acc_quat"].values).all()


def test_manifest_builder_is_label_free_and_records_provenance(tmp_path):
    sequence_dir = tmp_path / "sequences"
    sequence_dir.mkdir()
    skeleton = np.repeat(_skeleton(6)[:, None], 1, axis=1)
    imu = np.zeros((6, 1, 7), dtype=np.float64)
    imu[..., 0] = 1.0
    imu[..., 3] = 1.0
    np.savez_compressed(
        sequence_dir / "fixture.npz",
        dataset=np.array("totalcapture", dtype=object),
        sequence_id=np.array("fixture", dtype=object),
        frame_ids=np.arange(6),
        gt_skeleton=skeleton,
        imu=imu,
        imu_channels=np.asarray(["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"], dtype=object),
    )
    args = type("Args", (), {
        "root": str(tmp_path),
        "dataset": "totalcapture",
        "fps_hz": 30.0,
        "joint_layout": "coco17",
        "coordinate_space": "pixel",
        "sensor_location": "L_LowArm",
        "provenance": "xsens_measured",
        "channel_names": None,
        "max_sequences": None,
    })()
    manifest = build_manifest(args)
    assert manifest["sequence_count_succeeded"] == 1
    assert "correct" not in manifest and "frame_acc" not in manifest
    record = manifest["records"][0]
    assert record["sensor_location"] == "L_LowArm"
    assert record["persons"][0]["imu_views"]["I6_acc_quat"]["shape"] == [6, 7]


def test_physical_screen_skips_singletons_and_scores_multi_candidate_groups(tmp_path):
    sequence_dir = tmp_path / "sequences"
    sequence_dir.mkdir()
    skeleton = np.repeat(_skeleton(6)[:, None], 2, axis=1)
    imu = np.zeros((6, 2, 7), dtype=np.float64)
    imu[..., 0] = 1.0
    imu[..., 3] = 1.0
    np.savez_compressed(
        sequence_dir / "fixture.npz",
        gt_skeleton=skeleton,
        imu=imu,
        imu_channels=np.asarray(["acc_x", "acc_y", "acc_z", "quat_w", "quat_x", "quat_y", "quat_z"], dtype=object),
    )
    csv_path = tmp_path / "windows.csv"
    csv_path.write_text(
        "npz_path,window_start,window_end,person_idx,imu_idx\n"
        "sequences/fixture.npz,0,6,0,0\n"
        "sequences/fixture.npz,0,6,1,1\n"
        "sequences/fixture.npz,0,3,0,0\n",
        encoding="utf-8",
    )
    args = type("Args", (), {
        "dataset": "egohumans",
        "session_role": "source",
        "root": str(tmp_path),
        "csv": str(csv_path),
        "fps_hz": 30.0,
        "joint_layout": "coco17",
        "channel_names": None,
        "anchor_id": "A0_left_wrist",
        "skeleton_feature": "speed",
        "imu_view": "I1_acc_magnitude",
        "imu_feature": "magnitude",
        "comparator": "ncc",
        "max_lag": 1,
    })()
    result = run_screen(args)
    assert result["candidate_groups"] == 2
    assert result["singleton_groups_skipped"] == 1
    assert result["total"] == 2


def test_custom_gyro_parser_converts_units_and_preserves_time(tmp_path):
    path = tmp_path / "imu.csv"
    path.write_text(
        "epoch_ms,加速度X(g),加速度Y(g),加速度Z(g),角速度X(°/s),角速度Y(°/s),角速度Z(°/s)\n"
        "1000,1,0,0,180,0,0\n"
        "1050,0,1,0,0,90,0\n",
        encoding="utf-8",
    )
    record = parse_custom_csv(path)
    np.testing.assert_allclose(record.timestamps_s, [0.0, 0.05])
    np.testing.assert_allclose(record.acceleration_mps2[0], [9.80665, 0.0, 0.0])
    np.testing.assert_allclose(record.gyroscope_rads[0], [np.pi, 0.0, 0.0])
    assert record.sensor_location == "left_wrist"
    assert record.provenance == "custom_device_measured_csv"
