import numpy as np
import pytest

from src.features.orientation import (
    OrientationContractError,
    derive_2d_torso_proxy,
    derive_3d_torso_heading,
    direct_root_orientation,
)


def _skeleton_2d(t: int = 6) -> np.ndarray:
    skeleton = np.zeros((t, 17, 2), dtype=np.float64)
    angle = np.arange(t, dtype=np.float64) * 0.1
    skeleton[:, 11] = np.stack([-np.cos(angle), -np.sin(angle)], axis=-1)
    skeleton[:, 14] = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    return skeleton


def _skeleton_3d(t: int = 6) -> np.ndarray:
    skeleton = np.zeros((t, 17, 3), dtype=np.float64)
    angle = np.arange(t, dtype=np.float64) * 0.1
    lateral = np.stack([np.cos(angle), np.zeros(t), np.sin(angle)], axis=-1)
    skeleton[:, 11] = -lateral
    skeleton[:, 14] = lateral
    skeleton[:, 0] = 0.0
    skeleton[:, 8] = np.array([0.0, 1.0, 0.0])
    return skeleton


def test_2d_proxy_is_pi_periodic_and_uses_real_timestamps():
    result = derive_2d_torso_proxy(_skeleton_2d(), np.arange(6, dtype=np.float64) * 0.1)
    assert result.orientation_kind == "2d_proxy"
    assert result.angle_period == np.pi
    assert result.orientation_valid.all()
    assert result.rate_valid.all()
    np.testing.assert_allclose(result.angle_rate[1:-1], 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(result.direction, axis=-1), 1.0)


def test_2d_proxy_marks_missing_and_degenerate_frames():
    skeleton = _skeleton_2d(3)
    skeleton[1, 14] = skeleton[1, 11]
    visibility = np.ones((3, 17), dtype=bool)
    visibility[2, 11] = False
    result = derive_2d_torso_proxy(skeleton, [0.0, 0.1, 0.2], visibility=visibility)
    assert result.orientation_valid.tolist() == [True, False, False]
    assert result.degeneracy_reason[1] == "degenerate_shoulder_axis"
    assert result.degeneracy_reason[2] == "missing_shoulder"


def test_3d_heading_requires_explicit_up_axis_and_is_coordinate_dependent():
    result = derive_3d_torso_heading(_skeleton_3d(), np.arange(6, dtype=np.float64) * 0.1, up_axis=1)
    assert result.orientation_kind == "3d_derived"
    assert result.orientation_valid.all()
    assert result.coordinate_frame.endswith("up_y")
    np.testing.assert_allclose(np.linalg.norm(result.direction, axis=-1), 1.0, atol=1e-12)
    flipped = derive_3d_torso_heading(
        _skeleton_3d(), np.arange(6, dtype=np.float64) * 0.1, up_axis=1, cross_order="up_cross_lateral"
    )
    np.testing.assert_allclose(flipped.direction, -result.direction)


def test_direct_axis_angle_and_quaternion_sign_continuity():
    axis_angle = np.zeros((4, 3), dtype=np.float64)
    axis_angle[:, 1] = np.linspace(0.0, np.pi / 2.0, 4)
    direct = direct_root_orientation(np.arange(4, dtype=np.float64) * 0.2, axis_angle=axis_angle)
    assert direct.orientation_kind == "direct"
    assert direct.orientation_6d.shape == (4, 6)
    assert direct.orientation_valid.all()
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    quat_track = direct_root_orientation([0.0, 0.1, 0.2], quaternion=quaternions, quaternion_order="wxyz")
    np.testing.assert_allclose(quat_track.quaternion_continuous, np.array([[1.0, 0.0, 0.0, 0.0]] * 3))


def test_contract_rejects_nonmonotonic_timestamps_and_2d_padded_z():
    with pytest.raises(OrientationContractError, match="strictly increasing"):
        derive_2d_torso_proxy(_skeleton_2d(3), [0.0, 0.2, 0.1])
    with pytest.raises(OrientationContractError, match="2D proxy"):
        derive_2d_torso_proxy(np.zeros((3, 17, 3)), [0.0, 0.1, 0.2])
