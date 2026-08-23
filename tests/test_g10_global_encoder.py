from __future__ import annotations

import torch

from src.g10.global_encoder import (
    GlobalMotionMatcher,
    _frequency_profile,
    _identity,
    _window_key,
    _window_spectral_feature,
)
from tools.g10.train_global_encoder import _mmd_rbf_loss


def test_global_motion_matcher_uses_explicit_feature_widths() -> None:
    model = GlobalMotionMatcher(skeleton_dim=1, imu_dim=10, hidden=16, embedding_dim=8)
    output = model(torch.randn(4, 24, 1), torch.randn(4, 24, 10))
    assert output["skeleton"].shape == (4, 8)
    assert output["imu"].shape == (4, 8)
    assert torch.isfinite(output["skeleton"]).all()
    assert torch.isfinite(output["imu"]).all()
    assert torch.allclose(output["skeleton"].norm(dim=-1), torch.ones(4), atol=1e-5)


def test_transformer_temporal_mode_preserves_embedding_contract() -> None:
    model = GlobalMotionMatcher(skeleton_dim=2, imu_dim=3, hidden=16, embedding_dim=8, temporal_mode="transformer")
    output = model(torch.randn(3, 24, 2), torch.randn(3, 24, 3))
    assert output["skeleton"].shape == (3, 8)
    assert output["imu"].shape == (3, 8)
    assert torch.isfinite(output["skeleton"]).all()


def test_candidate_identity_does_not_use_constant_source_person() -> None:
    first = {
        "candidate_group_id": "cross:test:0",
        "candidate_index": "0",
        "source_person": "0",
        "source_sequence": "sequence_a",
    }
    second = {**first, "candidate_index": "1", "source_sequence": "sequence_b"}
    assert _window_key(first) == _window_key(second)
    assert _identity(first) != _identity(second)


def test_mmd_rbf_is_finite_and_nonnegative() -> None:
    first = torch.randn(8, 6)
    second = torch.randn(8, 6) + 0.5
    value = _mmd_rbf_loss(first, second)
    assert torch.isfinite(value)
    assert value >= 0


def test_window_spectral_feature_is_constant_and_finite() -> None:
    import numpy as np

    timestamps = np.arange(24, dtype=float) / 30.0
    signal = np.stack([np.sin(2 * np.pi * 2.0 * timestamps), np.zeros(24)], axis=-1)
    feature = _window_spectral_feature(signal, timestamps, np.ones(24, dtype=bool), "dominant_frequency")
    assert feature.shape == (24, 1)
    assert np.isfinite(feature).all()
    assert np.allclose(feature, feature[0])


def test_frequency_profile_has_common_band_width() -> None:
    import numpy as np

    timestamps = np.arange(24, dtype=float) / 30.0
    signal = np.stack([np.sin(2 * np.pi * 2.0 * timestamps), np.zeros(24)], axis=-1)
    profile = _frequency_profile(signal, timestamps, np.ones(24, dtype=bool), 16)
    assert profile.shape == (24, 16)
    assert np.isfinite(profile).all()
