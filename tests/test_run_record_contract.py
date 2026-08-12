from __future__ import annotations

import json
import subprocess

import pytest
import torch

from src.config import get_cfg_defaults
from src.experiments import build_evaluation_run_record


def _configured_cfg(tmp_path):
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.PREPROCESS.DATASET = "custom"
    cfg.TRAIN.SEED = 42
    cfg.EXPERIMENT.JOB_ID = "evaluate.direct.none.fold1.seed42"
    cfg.EXPERIMENT.TRAIN_JOB_ID = "train.direct.none.fold1.seed42"
    cfg.EXPERIMENT.PROTOCOL_HASH = "protocol-hash"
    cfg.EXPERIMENT.DATA_MANIFEST_HASH = "manifest-hash"
    cfg.EXPERIMENT.CONDITION = "direct"
    cfg.EXPERIMENT.TEST_SESSION = "20260211_171423"
    cfg.freeze()
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {}}, checkpoint)
    raw_results = tmp_path / "results.json"
    raw_results.write_text(json.dumps({"evaluations": {}}), encoding="utf-8")
    return cfg, checkpoint, raw_results


def test_run_record_contains_raw_counts_hashes_and_git_provenance(tmp_path):
    cfg, checkpoint, raw_results = _configured_cfg(tmp_path)
    output = {
        "evaluations": {
            "frame_acc": {
                "correct": 7,
                "total": 10,
                "frame_acc": 0.65,
                "weighted_frame_acc": 0.7,
            }
        }
    }

    record = build_evaluation_run_record(
        cfg,
        checkpoint=checkpoint,
        evaluation_output=output,
        raw_results_path=raw_results,
        repo_root=".",
    )

    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert record is not None
    assert record["train_job_id"] == "train.direct.none.fold1.seed42"
    assert record["frame_acc"] == {"correct": 7, "total": 10, "value": 0.7}
    assert record["git_commit"] == expected_commit
    assert record["runtime"]["python"]
    assert record["runtime"]["torch"] == torch.__version__
    assert "cuda_available" in record["runtime"]
    assert len(record["checkpoint_sha256"]) == 64
    assert len(record["raw_results_sha256"]) == 64
    assert len(record["config_hash"]) == 64


def test_run_record_requires_protocol_manifest_and_consistent_counts(tmp_path):
    cfg, checkpoint, raw_results = _configured_cfg(tmp_path)
    cfg.defrost()
    cfg.EXPERIMENT.PROTOCOL_HASH = ""
    cfg.freeze()
    with pytest.raises(ValueError, match="PROTOCOL_HASH"):
        build_evaluation_run_record(
            cfg,
            checkpoint=checkpoint,
            evaluation_output={
                "evaluations": {"frame_acc": {"correct": 1, "total": 2}}
            },
            raw_results_path=raw_results,
            repo_root=".",
        )


def test_run_record_is_optional_for_nonformal_evaluation(tmp_path):
    cfg = get_cfg_defaults()
    cfg.freeze()
    assert build_evaluation_run_record(
        cfg,
        checkpoint=tmp_path / "not-needed.pt",
        evaluation_output={},
        raw_results_path=tmp_path / "not-needed.json",
        repo_root=".",
    ) is None
