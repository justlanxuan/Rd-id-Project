from __future__ import annotations

import torch

from src.engine.batch import group_labels_from_batch, parse_domain_label_map, subject_labels_from_batch
from src.engine.losses import retrieval_top1


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
