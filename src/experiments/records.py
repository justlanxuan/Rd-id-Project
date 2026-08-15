"""Canonical evaluation run-record schema and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

RUN_RECORD_SCHEMA_VERSION = "1.0"
VALID_CONDITIONS = {"source", "zero_shot", "finetune", "direct"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _runtime_provenance() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_build": str(torch.version.cuda or ""),
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if cuda_available:
        device_index = int(torch.cuda.current_device())
        runtime["cuda_device_index"] = device_index
        runtime["cuda_device_name"] = str(torch.cuda.get_device_name(device_index))
    return runtime


def _primary_frame_acc(frame_result: dict[str, Any]) -> tuple[int, int, float]:
    if "correct" in frame_result and "total" in frame_result:
        correct = int(frame_result["correct"])
        total = int(frame_result["total"])
    elif "correct_assignments" in frame_result and "num_assignments" in frame_result:
        correct = int(frame_result["correct_assignments"])
        total = int(frame_result["num_assignments"])
    else:
        raise KeyError("FrameAcc result has no canonical correct/total counts.")
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError(f"Invalid FrameAcc counts: {correct}/{total}")
    return correct, total, float(correct / total)


def build_evaluation_run_record(
    cfg: Any,
    *,
    checkpoint: str | Path,
    evaluation_output: dict[str, Any],
    raw_results_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any] | None:
    experiment = cfg.EXPERIMENT
    job_id = str(experiment.JOB_ID).strip()
    if not job_id:
        return None

    protocol_hash = str(experiment.PROTOCOL_HASH).strip()
    data_manifest_hash = str(experiment.DATA_MANIFEST_HASH).strip()
    condition = str(experiment.CONDITION).strip()
    if not protocol_hash:
        raise ValueError("EXPERIMENT.PROTOCOL_HASH is required when JOB_ID is set.")
    if not data_manifest_hash:
        raise ValueError("EXPERIMENT.DATA_MANIFEST_HASH is required when JOB_ID is set.")
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"EXPERIMENT.CONDITION must be one of {sorted(VALID_CONDITIONS)}, got {condition!r}."
        )

    frame_result = evaluation_output.get("evaluations", {}).get("frame_acc")
    if not isinstance(frame_result, dict):
        raise ValueError("Evaluation output has no FrameAcc result for the canonical run record.")
    correct, total, value = _primary_frame_acc(frame_result)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
    raw_path = Path(raw_results_path).expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw evaluation results not found: {raw_path}")

    resolved_config = cfg.dump(sort_keys=True)
    config_hash = hashlib.sha256(resolved_config.encode("utf-8")).hexdigest()
    root = Path(repo_root).expanduser().resolve()
    git_commit, git_dirty = _git_provenance(root)
    source = str(experiment.SOURCE).strip() or None
    test_session = str(experiment.TEST_SESSION).strip() or None
    candidate_groups = {
        key: frame_result[key]
        for key in (
            "num_candidate_windows",
            "num_evaluated_windows",
            "num_singleton_windows",
            "singleton_rate",
            "candidate_group_size_min",
            "candidate_group_size_mean",
        )
        if key in frame_result
    }
    return {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "job_id": job_id,
        "train_job_id": str(experiment.TRAIN_JOB_ID).strip(),
        "status": "completed",
        "protocol_hash": protocol_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runtime": _runtime_provenance(),
        "config_hash": config_hash,
        "resolved_config": resolved_config,
        "data_manifest_hash": data_manifest_hash,
        "dataset": str(cfg.PREPROCESS.DATASET),
        "source": source,
        "condition": condition,
        "seed": int(cfg.TRAIN.SEED),
        "test_session": test_session,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "raw_results": str(raw_path),
        "raw_results_sha256": _sha256_file(raw_path),
        "frame_acc": {"correct": correct, "total": total, "value": value},
        "candidate_groups": candidate_groups,
    }


def write_evaluation_run_record(
    cfg: Any,
    *,
    checkpoint: str | Path,
    evaluation_output: dict[str, Any],
    raw_results_path: str | Path,
    default_output_path: str | Path,
    repo_root: str | Path,
) -> Path | None:
    record = build_evaluation_run_record(
        cfg,
        checkpoint=checkpoint,
        evaluation_output=evaluation_output,
        raw_results_path=raw_results_path,
        repo_root=repo_root,
    )
    if record is None:
        return None
    configured = str(cfg.EXPERIMENT.RUN_RECORD).strip()
    output = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(default_output_path).expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
