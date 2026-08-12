"""Create non-formal one-epoch smoke configs from a locked G6 job index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.config import load_cfg

SMOKE_TRAIN_JOBS = (
    "train.source.totalcapture.seed0",
    "train.direct.none.fold1.seed0",
)


def build_smoke_configs(
    formal_index: str | Path,
    output_dir: str | Path,
    *,
    artifact_root: str | Path,
    max_steps_per_epoch: int = 10,
) -> list[dict[str, Any]]:
    if max_steps_per_epoch <= 0:
        raise ValueError("Smoke max_steps_per_epoch must be positive.")
    index_path = Path(formal_index).expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    protocol_hash = str(index.get("protocol_hash", ""))
    if len(protocol_hash) != 64:
        raise ValueError("Smoke configs require a protocol-locked formal job index.")
    jobs = {str(row["job_id"]): row for row in index.get("jobs", [])}
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = Path(artifact_root).expanduser().resolve()
    entries = []
    for job_id in SMOKE_TRAIN_JOBS:
        if job_id not in jobs:
            raise ValueError(f"Formal index is missing smoke source job {job_id}")
        formal = jobs[job_id]
        source_path = (index_path.parent / str(formal["config"])).resolve()
        config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError(f"Formal config must be a mapping: {source_path}")
        smoke_name = f"smoke.{job_id}"
        safe_name = smoke_name.replace(".", "__")
        config["project"] = safe_name
        train = config.setdefault("train", {})
        train["epochs"] = 1
        train["max_steps_per_epoch"] = max_steps_per_epoch
        train["num_workers"] = 0
        train["early_stop_patience"] = 0
        train["output"] = {
            "output_root": str(artifacts / "train"),
            "run_name": safe_name,
        }
        test = config.setdefault("test", {})
        test["checkpoint"] = str(artifacts / "train" / safe_name / "best.pt")
        test["num_workers"] = 0
        test["output"] = {
            "output_root": str(artifacts / "evaluate"),
            "run_name": safe_name,
        }
        test.setdefault("metrics", {}).setdefault("group_test", {})["enabled"] = False
        experiment = config.setdefault("experiment", {})
        experiment["job_id"] = ""
        experiment["train_job_id"] = smoke_name
        experiment["run_record"] = ""
        filename = f"{safe_name}.yaml"
        path = destination / filename
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        resolved = load_cfg(path)
        if resolved.TRAIN.EPOCHS != 1 or resolved.TRAIN.MAX_STEPS_PER_EPOCH != max_steps_per_epoch:
            raise ValueError(f"Smoke budget did not resolve correctly: {path}")
        if resolved.EXPERIMENT.JOB_ID:
            raise ValueError(f"Smoke config must not emit a formal run record: {path}")
        entries.append(
            {
                "smoke_id": smoke_name,
                "source_job_id": job_id,
                "config": filename,
                "stages": "preprocess,train,test",
            }
        )
    return entries
