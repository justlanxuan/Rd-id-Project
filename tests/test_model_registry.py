from __future__ import annotations

import torch

from src.config import get_cfg_defaults
from src.models.checkpoint import (
    adapt_checkpoint_state,
    load_model_checkpoint,
    model_checkpoint_metadata,
)
from src.models.registry import MODEL_REGISTRY, build_model


def test_hybrid_model_is_built_through_model_registry_and_has_stable_output():
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.TRAIN.MODEL.HYBRID_HIDDEN = 32
    cfg.TRAIN.MODEL.HYBRID_TOKEN_HEADS = 4
    cfg.TRAIN.MODEL.HYBRID_TEMPORAL_LAYERS = 1
    cfg.freeze()

    model, name = build_model(cfg, torch.device("cpu"))
    output = model(
        imu=torch.zeros(2, 24, 7),
        skeleton=torch.zeros(2, 24, 17, 3),
    )

    assert MODEL_REGISTRY.names() == ("hybrid", "orientation_aware")
    assert name == "hybrid"
    assert set(output) == {"imu", "video"}
    assert output["imu"].shape == (2, 32)
    assert output["video"].shape == (2, 32)
    assert output["imu"].isfinite().all()
    assert output["video"].isfinite().all()
    assert model.capabilities.fitted_input_stats is True
    assert model.capabilities.full_validation_batch is True
    assert model.capabilities.segment_frame_acc is True
    assert model.capabilities.preferred_validation_metric == "val_loss"


def test_legacy_hybrid_checkpoint_keys_have_an_explicit_migration():
    state = {
        "skel.proj.weight": torch.zeros(2, 2),
        "imu.proj.weight": torch.ones(2, 2),
        "log_temp": torch.tensor(0.0),
    }

    migrated = adapt_checkpoint_state("hybrid", {"model": state})

    assert set(migrated) == {"video_encoder.proj.weight", "imu_encoder.raw.proj.weight"}


def test_hybrid_checkpoint_adapter_owns_legacy_stats_and_shape_filtering():
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.TRAIN.MODEL.HYBRID_HIDDEN = 32
    cfg.TRAIN.MODEL.HYBRID_TOKEN_HEADS = 4
    cfg.TRAIN.MODEL.HYBRID_TEMPORAL_LAYERS = 1
    cfg.freeze()
    model, name = build_model(cfg, torch.device("cpu"))
    payload = {
        "model": {
            "imu.proj.weight": torch.ones(1, 1),
            "log_temp": torch.tensor(0.0),
        },
        "stats": {"imu_mu": torch.zeros_like(model.state_dict()["imu_encoder.imu_mu"])},
    }

    adapted = adapt_checkpoint_state(name, payload)
    report = load_model_checkpoint(
        model,
        name,
        payload,
        allow_shape_mismatch=True,
    )

    assert "imu_encoder.imu_mu" in adapted
    assert "log_temp" not in adapted
    assert report.checkpoint == "<in-memory>"
    assert report.dropped_incompatible_keys == ("imu_encoder.raw.proj.weight",)


def test_versioned_checkpoint_metadata_rejects_wrong_model_name():
    cfg = get_cfg_defaults()
    cfg.freeze()
    model, name = build_model(cfg, torch.device("cpu"))
    metadata = model_checkpoint_metadata(name, model)

    assert metadata["checkpoint_schema_version"] == "1.0"
    assert metadata["model_name"] == "hybrid"
    assert metadata["model_capabilities"]["segment_frame_acc"] is True
    payload = {**metadata, "model_name": "another_model", "model": model.state_dict()}
    try:
        load_model_checkpoint(model, name, payload)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("A checkpoint for another model type must be rejected.")


def test_versioned_checkpoint_round_trip_preserves_model_output(tmp_path):
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.TRAIN.MODEL.HYBRID_HIDDEN = 32
    cfg.TRAIN.MODEL.HYBRID_TOKEN_HEADS = 4
    cfg.TRAIN.MODEL.HYBRID_TEMPORAL_LAYERS = 1
    cfg.TRAIN.MODEL.HYBRID_DROPOUT = 0.0
    cfg.freeze()
    source, name = build_model(cfg, torch.device("cpu"))
    source.eval()
    imu = torch.randn(2, 24, 7)
    skeleton = torch.randn(2, 24, 17, 3)
    with torch.no_grad():
        expected = source(imu=imu, skeleton=skeleton)
    checkpoint = tmp_path / "round-trip.pt"
    torch.save({**model_checkpoint_metadata(name, source), "model": source.state_dict()}, checkpoint)

    restored, restored_name = build_model(cfg, torch.device("cpu"))
    report = load_model_checkpoint(restored, restored_name, checkpoint, strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(imu=imu, skeleton=skeleton)

    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert report.dropped_incompatible_keys == ()
    assert torch.equal(actual["imu"], expected["imu"])
    assert torch.equal(actual["video"], expected["video"])
