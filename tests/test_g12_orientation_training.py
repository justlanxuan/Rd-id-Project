import numpy as np
import torch

from src.g12.orientation_matcher import OrientationAwareMatcher
from tools.g10.train_global_encoder import _info_nce
from tools.g12.evaluate_physical_turning_moe import _physical_score
from tools.g12.train_orientation_matcher import (
    OrientationHardNegativeBatchSampler,
    _turn_onset_loss,
    _weighted_info_nce,
    evaluate,
)


class _FakeOrientationDataset:
    def __init__(self) -> None:
        self.rows = []
        self.activity = []
        for session, offset in (("acting", 0.0), ("walking", 0.5)):
            for index in range(8):
                self.rows.append({"session": session, "_dataset": "tc", "_group_key": f"{session}-{index}"})
                self.activity.append(offset + index * 0.01)

    def group_indices(self):
        return {(row["_dataset"], row["_group_key"]): [index] for index, row in enumerate(self.rows)}

    def __getitem__(self, index):
        orientation = torch.zeros(4, 5)
        orientation[:, 4] = self.activity[index]
        return {"orientation": orientation}


def test_weighted_info_nce_matches_unweighted_for_uniform_weights():
    generator = torch.Generator().manual_seed(7)
    output = {
        "imu": torch.nn.functional.normalize(torch.randn(6, 8, generator=generator), dim=-1),
        "skeleton": torch.nn.functional.normalize(torch.randn(6, 8, generator=generator), dim=-1),
    }
    expected, expected_accuracy = _info_nce(output, 0.1)
    actual, actual_accuracy = _weighted_info_nce(output, 0.1, torch.ones(6))
    torch.testing.assert_close(actual, expected)
    assert actual_accuracy == expected_accuracy


def test_orientation_hard_negative_sampler_keeps_each_batch_in_one_action():
    dataset = _FakeOrientationDataset()
    sampler = OrientationHardNegativeBatchSampler(dataset, batch_size=4, seed=3, steps=4, pool_multiplier=2)
    batches = list(sampler)
    assert len(batches) == 4
    for batch in batches:
        assert len(batch) == 4
        assert len({dataset.rows[index]["session"] for index in batch}) == 1
        values = np.asarray([dataset.activity[index] for index in batch])
        assert float(values.max() - values.min()) <= 0.07


def test_conditional_cross_preserves_dual_tower_modality_isolation():
    torch.manual_seed(11)
    model = OrientationAwareMatcher(
        skeleton_dim=9,
        imu_dim=6,
        orientation_dim=5,
        hidden=12,
        embedding_dim=8,
        temporal_mode="gru",
        fusion_mode="conditional_cross",
    ).eval()
    skeleton = torch.randn(2, 6, 9)
    imu = torch.randn(2, 6, 6)
    orientation = torch.randn(2, 6, 5)
    orientation[..., 4] = torch.rand(2, 6)
    original = model(skeleton, imu, orientation)
    changed_skeleton = model(skeleton + 10.0, imu, 1.0 - orientation)
    changed_imu = model(skeleton, imu - 10.0, orientation)
    torch.testing.assert_close(original["imu"], changed_skeleton["imu"])
    torch.testing.assert_close(original["skeleton"], changed_imu["skeleton"])
    torch.testing.assert_close(original["gyro_onset_logits"], changed_skeleton["gyro_onset_logits"])
    torch.testing.assert_close(original["orientation_onset_logits"], changed_imu["orientation_onset_logits"])
    loss = _turn_onset_loss(original, orientation)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_physical_turning_score_prefers_aligned_rate_and_gyro_activity():
    rate = np.asarray([0.1, 0.8, 0.2, 0.4, 1.0, 0.3, 0.7, 0.05], dtype=np.float32)
    orientation = np.zeros((len(rate), 5), dtype=np.float32)
    orientation[:, 2] = rate
    aligned = np.zeros((len(rate), 6), dtype=np.float32)
    aligned[:, 3] = rate
    shuffled = aligned.copy()
    shuffled[:, 3] = np.asarray([0.4, 0.1, 0.7, 1.0, 0.05, 0.8, 0.3, 0.2])
    assert _physical_score(orientation, aligned) > _physical_score(orientation, shuffled)


def test_evaluate_assigns_turn_stratum_at_candidate_group_level():
    class Dataset:
        rows = [
            {"_dataset": "custom23", "_group_key": group, "_identity": str(identity)}
            for group in ("first", "second")
            for identity in range(2)
        ]

        def __len__(self):
            return 4

        def __getitem__(self, index):
            identity = index % 2
            embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])[identity].repeat(2, 1)
            orientation = torch.zeros(2, 5)
            orientation[:, 4] = 1.0 if index == 0 else 0.0
            return {
                "skeleton": embedding,
                "imu": embedding,
                "orientation": orientation,
                "domain": "custom23",
                "group_key": self.rows[index]["_group_key"],
                "identity": self.rows[index]["_identity"],
            }

    class Model(torch.nn.Module):
        def forward(self, skeleton, imu, orientation):
            return {"skeleton": skeleton[:, 0], "imu": imu[:, 0]}

    result = evaluate(Model(), Dataset(), torch.device("cpu"), turning_threshold=0.5)
    assert result["turning_strata"]["custom23"]["high"]["total"] == 2
    assert result["turning_strata"]["custom23"]["low"]["total"] == 2
