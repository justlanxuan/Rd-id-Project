from __future__ import annotations

import pytest
import torch

from src.g10.global_encoder import GlobalMotionMatcher
from src.modules.encoders.multiscale import (
    DEFAULT_DILATIONS,
    HierarchicalTemporalAttention,
    MultiScaleTemporalTCN,
    receptive_field_samples,
    receptive_field_seconds,
)


@pytest.mark.parametrize("fusion", ["mean", "gated", "hierarchical_attention"])
def test_multiscale_tcn_preserves_sequence_and_returns_auditable_weights(fusion: str) -> None:
    model = MultiScaleTemporalTCN(input_dim=5, hidden_dim=12, output_dim=7, fusion=fusion, dropout=0.0)
    output = model(torch.randn(3, 30, 5), fps_hz=30.0, window_seconds=1.0)
    assert output["sequence"].shape == (3, 30, 7)
    assert output["branches"].shape == (3, 3, 30, 12)
    assert output["scale_weights"].shape == (3, 3)
    assert torch.isfinite(output["sequence"]).all()
    assert torch.isfinite(output["scale_weights"]).all()
    assert torch.allclose(output["scale_weights"].sum(dim=1), torch.ones(3), atol=1e-6)


def test_hierarchical_attention_respects_mask_and_all_masked_rows() -> None:
    attention = HierarchicalTemporalAttention(hidden_dim=4, mode="hierarchical_attention")
    scales = torch.randn(2, 3, 6, 4)
    mask = torch.tensor([[True, True, True, False, False, False], [False] * 6])
    fused, weights = attention(scales, mask)
    assert fused.shape == (2, 6, 4)
    assert weights.shape == (2, 3)
    assert torch.isfinite(fused).all()
    assert torch.isfinite(weights).all()
    assert torch.allclose(fused[0, 3:], torch.zeros(3, 4), atol=1e-6)
    assert torch.allclose(fused[1], torch.zeros(6, 4), atol=1e-6)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_temporal_block_backpropagates_finite_nonzero_gradients() -> None:
    model = MultiScaleTemporalTCN(input_dim=3, hidden_dim=8, fusion="gated", dropout=0.0)
    output = model(torch.randn(2, 20, 3))["sequence"]
    loss = output.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(torch.any(gradient.abs() > 0)) for gradient in gradients if gradient is not None)


def test_receptive_field_is_seconds_aware_and_profile_is_serializable() -> None:
    assert receptive_field_samples(3, (1, 2, 4)) == 15
    assert receptive_field_seconds(3, (1, 2, 4), 30.0) == pytest.approx(0.5)
    model = MultiScaleTemporalTCN(input_dim=2, hidden_dim=8, dropout=0.0)
    profile = model.profile_spec(30.0)
    assert profile["receptive_fields"]["short"]["samples"] == 15
    assert profile["receptive_fields"]["long"]["seconds"] == pytest.approx(141 / 30.0)
    assert profile["parameters"] > 0
    assert set(profile["dilations"]) == set(DEFAULT_DILATIONS)


def test_context_longer_than_ten_seconds_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        MultiScaleTemporalTCN(input_dim=2, window_seconds=10.01)
    model = MultiScaleTemporalTCN(input_dim=2)
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.randn(1, 301, 2), fps_hz=30.0, window_seconds=10.0)


def test_nonfinite_input_and_invalid_dilation_are_explicit_failures() -> None:
    model = MultiScaleTemporalTCN(input_dim=2)
    bad = torch.zeros(1, 8, 2)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        model(bad)
    with pytest.raises(ValueError, match="positive odd"):
        MultiScaleTemporalTCN(input_dim=2, kernel_size=2)
    with pytest.raises(ValueError, match="positive integers"):
        MultiScaleTemporalTCN(input_dim=2, dilations={"short": (0,), "middle": (1,), "long": (1,)})


def test_g10_global_matcher_multiscale_mode_keeps_embedding_and_weight_contract() -> None:
    model = GlobalMotionMatcher(
        skeleton_dim=2,
        imu_dim=3,
        hidden=16,
        embedding_dim=8,
        temporal_mode="multiscale",
        multiscale_fusion="hierarchical_attention",
        window_seconds=0.8,
    )
    output = model(torch.randn(4, 24, 2), torch.randn(4, 24, 3))
    assert output["skeleton"].shape == (4, 8)
    assert output["imu"].shape == (4, 8)
    assert output["skeleton_scale_weights"].shape == (4, 3)
    assert output["imu_scale_weights"].shape == (4, 3)
    assert torch.allclose(output["skeleton"].norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(output["imu_scale_weights"].sum(dim=-1), torch.ones(4), atol=1e-6)


def test_g10_global_matcher_tcn_control_has_same_embedding_contract() -> None:
    model = GlobalMotionMatcher(skeleton_dim=2, imu_dim=3, hidden=16, embedding_dim=8, temporal_mode="tcn")
    output = model(torch.randn(2, 24, 2), torch.randn(2, 24, 3))
    assert output["skeleton"].shape == (2, 8)
    assert output["imu"].shape == (2, 8)
    assert "skeleton_scale_weights" not in output
