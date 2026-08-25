import torch

from src.modules.encoders.hybrid import (
    H36M_FEATURE_MODES,
    h36m_feature_dim,
    h36m_feature_sequence,
)


def test_g13_3d_feature_modes_have_finite_contracts():
    skeleton = torch.randn(2, 24, 17, 3)
    for mode in H36M_FEATURE_MODES:
        features = h36m_feature_sequence(skeleton, smooth_kernel=3, feature_mode=mode)
        assert features.shape == (2, 24, h36m_feature_dim(mode))
        assert torch.isfinite(features).all()


def test_g13_zonly_removes_image_plane_coordinates():
    skeleton = torch.randn(1, 24, 17, 3)
    features = h36m_feature_sequence(skeleton, smooth_kernel=1, feature_mode="h36m3d_zonly")
    assert torch.allclose(features[..., : 17 * 3 : 3], torch.zeros_like(features[..., : 17 * 3 : 3]))
    assert torch.allclose(features[..., 1 : 17 * 3 : 3], torch.zeros_like(features[..., 1 : 17 * 3 : 3]))


def test_g13_3d_depth_changes_controlled_input():
    skeleton = torch.randn(1, 24, 17, 3)
    skeleton[..., 2] += 3.0
    control = h36m_feature_sequence(skeleton, smooth_kernel=1, feature_mode="h36m2d")
    full = h36m_feature_sequence(skeleton, smooth_kernel=1, feature_mode="h36m3d")
    assert not torch.allclose(control, full)
