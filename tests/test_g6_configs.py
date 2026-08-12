from __future__ import annotations

import json

from src.config import load_cfg
from tools.g6.configs import generate_resolved_configs


def test_generate_all_resolved_configs_with_dependency_checkpoints(tmp_path):
    manifests = [
        {"dataset": "totalcapture", "fold_id": None, "manifest_hash": "tc-hash"},
        {"dataset": "egohumans", "fold_id": None, "manifest_hash": "ego-hash"},
    ]
    manifests.extend(
        {"dataset": "custom", "fold_id": fold_id, "manifest_hash": f"custom-{fold_id}"}
        for fold_id in range(1, 5)
    )
    index = tmp_path / "data-index.json"
    index.write_text(json.dumps({"manifests": manifests}), encoding="utf-8")
    output_dir = tmp_path / "configs"
    artifact_root = tmp_path / "artifacts"

    rows = generate_resolved_configs(
        output_dir,
        protocol_hash="locked-protocol",
        data_manifest_index=index,
        artifact_root=artifact_root,
    )

    assert len(rows) == 108
    assert sum(row["job_type"] == "train" for row in rows) == 42
    assert sum(row["job_type"] == "evaluate" for row in rows) == 66
    by_id = {row["job_id"]: row for row in rows}
    for row in rows:
        cfg = load_cfg(output_dir / row["config"])
        if row["job_type"] == "train":
            assert cfg.EXPERIMENT.JOB_ID == ""
            assert cfg.EXPERIMENT.TRAIN_JOB_ID == row["job_id"]
            assert cfg.EXPERIMENT.PROTOCOL_HASH == "locked-protocol"
            assert row["stages"] == "preprocess,train"
            continue
        dependency = by_id[row["depends_on"]]
        expected_checkpoint = artifact_root / "train" / dependency["job_id"].replace(".", "__") / "best.pt"
        assert cfg.EXPERIMENT.JOB_ID == row["job_id"]
        assert cfg.EXPERIMENT.TRAIN_JOB_ID == row["depends_on"]
        assert cfg.EXPERIMENT.PROTOCOL_HASH == "locked-protocol"
        assert cfg.EXPERIMENT.TEST_SESSION == (row["test_session"] or "")
        assert cfg.TEST.CHECKPOINT == str(expected_checkpoint)
        assert row["stages"] == "test"
