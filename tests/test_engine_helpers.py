from __future__ import annotations

import pytest
import torch

from src.engine.augmentation import maybe_augment_inputs
from src.engine.batch import group_labels_from_batch, parse_domain_label_map, subject_labels_from_batch
from src.engine.losses import retrieval_top1
from src.engine.train import require_finite_metrics, require_finite_tensor
from src.modules.encoders.hybrid import LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, skeleton_tokens


def test_parse_domain_label_map():
    assert parse_domain_label_map("egohumans:0, custom:1") == {"egohumans": 0, "custom": 1}


def test_subject_and_group_labels_from_batch():
    device = torch.device("cpu")
    subject_labels = subject_labels_from_batch(["S2", "S1", "S2"], {"S1": 0, "S2": 1}, device)
    group_labels = group_labels_from_batch(["a", "b", "a"], device)

    assert subject_labels.tolist() == [1, 0, 1]
    assert group_labels.tolist() == [0, 1, 0]


def test_retrieval_top1_identity_embeddings():
    z = torch.eye(4)
    assert retrieval_top1(z, z) == 1.0


def test_skeleton_tokens_stay_bounded_when_joint_dropout_collapses_forearm():
    skeleton = torch.ones(2, 24, 17, 3, requires_grad=True)
    skeleton.data[:, :, LEFT_SHOULDER, :2] = torch.tensor([100.0, 100.0])
    skeleton.data[:, :, LEFT_ELBOW, :2] = torch.tensor([200.0, 100.0])
    skeleton.data[:, :, LEFT_WRIST, :2] = 0.0

    tokens = skeleton_tokens(skeleton)
    loss = tokens.square().mean()
    loss.backward()

    assert torch.isfinite(tokens).all()
    assert tokens.abs().max() < 100
    assert torch.isfinite(skeleton.grad).all()


def test_training_finite_guards_reject_nan_before_artifact_write():
    with pytest.raises(FloatingPointError, match="training loss"):
        require_finite_tensor(torch.tensor(float("nan")), name="training loss", epoch=1, step=8)
    with pytest.raises(FloatingPointError, match="validation metrics"):
        require_finite_metrics({"loss": float("nan"), "top1": 0.5}, epoch=1)


def test_orientation_flattened_skeleton_supports_joint_dropout():
    cfg = type("Cfg", (), {})()
    cfg.TRAIN = type("Train", (), {
        "IMU_NOISE_STD": 0.0,
        "IMU_DROPOUT_PROB": 0.0,
        "SKEL_NOISE_STD": 0.0,
        "JOINT_DROPOUT_PROB": 0.5,
    })()
    imu = torch.ones(2, 24, 6)
    skeleton = torch.ones(2, 24, 51)
    _, augmented = maybe_augment_inputs(imu, skeleton, cfg)
    assert augmented.shape == skeleton.shape
    assert torch.isfinite(augmented).all()
