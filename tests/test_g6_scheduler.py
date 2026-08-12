from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config import load_cfg
from tools.g6.configs import generate_resolved_configs
from tools.g6.scheduler import (
    build_execution_plan,
    load_job_index,
    summarize_execution_plan,
)


def _generated_index(tmp_path: Path):
    manifests = [
        {"dataset": "totalcapture", "fold_id": None, "manifest_hash": "tc-hash"},
        {"dataset": "egohumans", "fold_id": None, "manifest_hash": "ego-hash"},
    ]
    manifests.extend(
        {"dataset": "custom", "fold_id": fold, "manifest_hash": f"custom-{fold}"}
        for fold in range(1, 5)
    )
    data_index = tmp_path / "data-index.json"
    data_index.write_text(json.dumps({"manifests": manifests}), encoding="utf-8")
    config_root = tmp_path / "configs"
    rows = generate_resolved_configs(
        config_root,
        protocol_hash="a" * 64,
        data_manifest_index=data_index,
        artifact_root=tmp_path / "artifacts",
    )
    index = config_root / "index.json"
    index.write_text(
        json.dumps({"protocol_hash": "a" * 64, "jobs": rows}),
        encoding="utf-8",
    )
    return index


def test_scheduler_dry_plan_has_only_dependency_free_training_jobs_ready(tmp_path):
    index = _generated_index(tmp_path)
    _, jobs = load_job_index(
        index,
        protocol_record={"status": "locked", "protocol_hash": "a" * 64},
    )

    statuses = build_execution_plan(jobs)
    summary = summarize_execution_plan(statuses)

    assert summary == {"pending": 90, "ready": 18}
    ready = {job_id for job_id, status in statuses.items() if status.status == "ready"}
    assert all(job_id.startswith("train.") for job_id in ready)
    assert sum("train.source" in job_id for job_id in ready) == 6
    assert sum("train.direct" in job_id for job_id in ready) == 12


def test_verified_source_checkpoint_unlocks_only_its_direct_dependents(tmp_path):
    index = _generated_index(tmp_path)
    _, jobs = load_job_index(
        index,
        protocol_record={"status": "locked", "protocol_hash": "a" * 64},
    )
    source_job = next(row for row in jobs if row["job_id"] == "train.source.totalcapture.seed0")
    cfg = load_cfg(source_job["_config_path"])
    checkpoint = Path(cfg.TEST.CHECKPOINT)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_schema_version": "1.0",
            "model_name": "hybrid",
            "model_capabilities": {},
            "epoch": 1,
            "config": cfg.dump(sort_keys=False),
            "model": {},
        },
        checkpoint,
    )
    (checkpoint.parent / "metrics.json").write_text(
        json.dumps({"stopped_epoch": 1, "save_dir": str(checkpoint.parent)}),
        encoding="utf-8",
    )

    statuses = build_execution_plan(jobs)
    summary = summarize_execution_plan(statuses)

    assert summary == {"completed": 1, "pending": 81, "ready": 26}
    assert statuses["evaluate.source.totalcapture.seed0"].status == "ready"
    assert statuses["train.finetune.totalcapture.fold1.seed0"].status == "ready"
    assert statuses["train.finetune.egohumans.fold1.seed0"].status == "pending"
