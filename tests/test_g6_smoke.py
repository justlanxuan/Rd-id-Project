from __future__ import annotations

import json

from src.config import load_cfg
from tools.g6.configs import generate_resolved_configs
from tools.g6.smoke import build_smoke_configs


def test_smoke_configs_are_bounded_and_cannot_emit_formal_records(tmp_path):
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
    formal_dir = tmp_path / "formal"
    formal_jobs = generate_resolved_configs(
        formal_dir,
        protocol_hash="b" * 64,
        data_manifest_index=data_index,
        artifact_root=tmp_path / "formal-artifacts",
    )
    formal_index = formal_dir / "index.json"
    formal_index.write_text(
        json.dumps({"protocol_hash": "b" * 64, "jobs": formal_jobs}),
        encoding="utf-8",
    )

    smoke_dir = tmp_path / "smoke"
    rows = build_smoke_configs(
        formal_index,
        smoke_dir,
        artifact_root=tmp_path / "smoke-artifacts",
        max_steps_per_epoch=3,
    )

    assert len(rows) == 2
    for row in rows:
        cfg = load_cfg(smoke_dir / row["config"])
        assert cfg.TRAIN.EPOCHS == 1
        assert cfg.TRAIN.MAX_STEPS_PER_EPOCH == 3
        assert cfg.EXPERIMENT.JOB_ID == ""
        assert cfg.EXPERIMENT.TRAIN_JOB_ID.startswith("smoke.train.")
        assert cfg.TEST.METRICS.GROUP_TEST.ENABLED is False
