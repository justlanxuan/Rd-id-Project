"""Dependency-aware, resumable execution support for protocol-locked G6 jobs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.config import load_cfg
from src.models.checkpoint import load_checkpoint_payload

from .git_snapshot import require_clean_git_snapshot
from .matrix import build_required_cells

REPO_ROOT = Path(__file__).resolve().parents[2]

JobStatus = Literal[
    "pending",
    "ready",
    "running",
    "completed",
    "failed",
    "blocked_dependency",
    "invalid_artifact",
]


@dataclass(frozen=True)
class JobArtifactStatus:
    status: JobStatus
    detail: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol_record(path: str | Path) -> dict[str, Any]:
    record_path = Path(path).expanduser().resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "locked":
        raise ValueError(f"Protocol record is not locked: {record_path}")
    protocol_hash = str(record.get("protocol_hash", "")).strip()
    if len(protocol_hash) != 64:
        raise ValueError(f"Protocol record has an invalid hash: {record_path}")
    git_commit = str(record.get("git_commit", "")).strip().lower()
    if len(git_commit) not in {40, 64} or any(char not in "0123456789abcdef" for char in git_commit):
        raise ValueError(f"Protocol record has no Git commit: {record_path}")
    return record


def load_job_index(
    path: str | Path,
    *,
    protocol_record: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    if protocol_record.get("status") != "locked":
        raise ValueError("Job index requires a locked protocol record.")
    index_path = Path(path).expanduser().resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    protocol_hash = str(payload.get("protocol_hash", ""))
    if protocol_hash != str(protocol_record["protocol_hash"]):
        raise ValueError(
            f"Job index protocol hash {protocol_hash!r} does not match locked record "
            f"{protocol_record['protocol_hash']!r}."
        )
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("Job index must contain a jobs list.")
    expected = {cell.job_id: cell.to_dict() for cell in build_required_cells()}
    by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise TypeError("Every indexed job must be an object.")
        job_id = str(job.get("job_id", ""))
        if job_id not in expected:
            raise ValueError(f"Unexpected G6 job id: {job_id!r}")
        if job_id in by_id:
            raise ValueError(f"Duplicate G6 job id: {job_id}")
        for key, value in expected[job_id].items():
            actual = job.get(key)
            if isinstance(value, tuple):
                actual = tuple(actual or ())
            if actual != value:
                raise ValueError(
                    f"Job index field mismatch for {job_id}.{key}: {actual!r} != {value!r}"
                )
        config_path = (index_path.parent / str(job.get("config", ""))).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Indexed config is missing: {config_path}")
        cfg = load_cfg(config_path)
        if str(cfg.EXPERIMENT.PROTOCOL_HASH) != protocol_hash:
            raise ValueError(f"Config protocol hash mismatch: {config_path}")
        if job["job_type"] == "train":
            if str(cfg.EXPERIMENT.TRAIN_JOB_ID) != job_id or str(cfg.EXPERIMENT.JOB_ID):
                raise ValueError(f"Training config job identity mismatch: {config_path}")
        elif str(cfg.EXPERIMENT.JOB_ID) != job_id:
            raise ValueError(f"Evaluation config job identity mismatch: {config_path}")
        row = dict(job)
        row["_config_path"] = str(config_path)
        by_id[job_id] = row
    missing = sorted(set(expected) - set(by_id))
    if missing:
        raise ValueError(f"Job index is incomplete; first missing jobs: {missing[:3]}")
    return index_path, [by_id[cell.job_id] for cell in build_required_cells()]


def _training_artifact_status(job: dict[str, Any]) -> JobArtifactStatus:
    cfg = load_cfg(job["_config_path"])
    checkpoint = Path(str(cfg.TEST.CHECKPOINT)).expanduser().resolve()
    metrics_path = checkpoint.parent / "metrics.json"
    present = [path for path in (checkpoint, metrics_path) if path.exists()]
    if not present:
        return JobArtifactStatus("pending")
    if len(present) != 2 or checkpoint.stat().st_size == 0:
        return JobArtifactStatus(
            "invalid_artifact",
            f"Partial training artifacts under {checkpoint.parent}",
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or "stopped_epoch" not in metrics:
            raise ValueError("metrics.json has no stopped_epoch")
        payload = load_checkpoint_payload(checkpoint)
        required = {
            "checkpoint_schema_version",
            "model_name",
            "model_capabilities",
            "epoch",
            "config",
            "model",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"checkpoint missing keys {missing}")
        embedded = yaml.safe_load(str(payload["config"]))
        experiment = embedded.get("EXPERIMENT", {}) if isinstance(embedded, dict) else {}
        if str(experiment.get("PROTOCOL_HASH", "")) != str(cfg.EXPERIMENT.PROTOCOL_HASH):
            raise ValueError("checkpoint protocol hash does not match config")
        if str(experiment.get("TRAIN_JOB_ID", "")) != str(job["job_id"]):
            raise ValueError("checkpoint train job id does not match index")
    except Exception as exc:
        return JobArtifactStatus("invalid_artifact", f"Invalid training artifact: {exc}")
    return JobArtifactStatus("completed", str(checkpoint))


def _evaluation_artifact_status(job: dict[str, Any]) -> JobArtifactStatus:
    cfg = load_cfg(job["_config_path"])
    record_path = Path(str(cfg.EXPERIMENT.RUN_RECORD)).expanduser().resolve()
    if not record_path.exists():
        return JobArtifactStatus("pending")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected = {
            "job_id": str(job["job_id"]),
            "train_job_id": str(job["depends_on"]),
            "protocol_hash": str(cfg.EXPERIMENT.PROTOCOL_HASH),
            "data_manifest_hash": str(cfg.EXPERIMENT.DATA_MANIFEST_HASH),
            "status": "completed",
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"{key} mismatch: {record.get(key)!r} != {value!r}")
        raw_results = Path(str(record["raw_results"])).expanduser().resolve()
        checkpoint = Path(str(record["checkpoint"])).expanduser().resolve()
        if not raw_results.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("referenced raw results or checkpoint is missing")
        if _sha256_file(raw_results) != str(record.get("raw_results_sha256", "")):
            raise ValueError("raw results SHA256 mismatch")
        if _sha256_file(checkpoint) != str(record.get("checkpoint_sha256", "")):
            raise ValueError("checkpoint SHA256 mismatch")
        frame_acc = record.get("frame_acc", {})
        if not isinstance(record.get("runtime"), dict):
            raise ValueError("run record has no runtime provenance")
        correct = int(frame_acc.get("correct", -1))
        total = int(frame_acc.get("total", 0))
        value = float(frame_acc.get("value", -1))
        if total <= 0 or correct < 0 or correct > total or abs(value - correct / total) > 1e-12:
            raise ValueError("invalid FrameAcc counts")
    except Exception as exc:
        return JobArtifactStatus("invalid_artifact", f"Invalid evaluation artifact: {exc}")
    return JobArtifactStatus("completed", str(record_path))


def artifact_status(job: dict[str, Any]) -> JobArtifactStatus:
    if job["job_type"] == "train":
        return _training_artifact_status(job)
    return _evaluation_artifact_status(job)


def build_execution_plan(jobs: list[dict[str, Any]]) -> dict[str, JobArtifactStatus]:
    statuses = {str(job["job_id"]): artifact_status(job) for job in jobs}
    _apply_dependency_states(jobs, statuses)
    return statuses


def _apply_dependency_states(
    jobs: list[dict[str, Any]],
    statuses: dict[str, JobArtifactStatus],
) -> None:
    by_id = {str(job["job_id"]): job for job in jobs}
    changed = True
    while changed:
        changed = False
        for job_id, job in by_id.items():
            if statuses[job_id].status != "pending":
                continue
            dependency = job.get("depends_on")
            if not dependency:
                statuses[job_id] = JobArtifactStatus("ready")
                changed = True
                continue
            dependency_status = statuses[str(dependency)]
            if dependency_status.status == "completed":
                statuses[job_id] = JobArtifactStatus("ready")
                changed = True
            elif dependency_status.status in {"failed", "invalid_artifact", "blocked_dependency"}:
                statuses[job_id] = JobArtifactStatus(
                    "blocked_dependency", f"Dependency {dependency}: {dependency_status.status}"
                )
                changed = True


def summarize_execution_plan(statuses: dict[str, JobArtifactStatus]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for value in statuses.values():
        summary[value.status] = summary.get(value.status, 0) + 1
    return dict(sorted(summary.items()))


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_state(
    path: Path,
    statuses: dict[str, JobArtifactStatus],
    running: dict[str, tuple[subprocess.Popen, object, str, Path]],
) -> None:
    payload = {
        "schema_version": "1.0",
        "summary": summarize_execution_plan(statuses),
        "jobs": {
            job_id: {
                "status": status.status,
                "detail": status.detail,
                "gpu": running[job_id][2] if job_id in running else None,
                "log": str(running[job_id][3]) if job_id in running else None,
            }
            for job_id, status in sorted(statuses.items())
        },
    }
    _atomic_write_json(path, payload)


def run_jobs(
    jobs: list[dict[str, Any]],
    *,
    gpus: list[str],
    max_parallel: int,
    log_root: str | Path,
    state_path: str | Path,
    dry_run: bool = False,
    poll_seconds: float = 5.0,
    expected_git_commit: str = "",
) -> dict[str, JobArtifactStatus]:
    if not gpus:
        raise ValueError("At least one explicit GPU id is required.")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU ids must be unique: {gpus}")
    parallel = max(1, min(int(max_parallel), len(gpus)))
    logs = Path(log_root).expanduser().resolve()
    state = Path(state_path).expanduser().resolve()
    statuses = build_execution_plan(jobs)
    invalid = {key: value for key, value in statuses.items() if value.status == "invalid_artifact"}
    if invalid:
        details = "; ".join(f"{key}: {value.detail}" for key, value in sorted(invalid.items()))
        raise RuntimeError(f"Refusing to overwrite invalid existing artifacts: {details}")

    if dry_run:
        _write_state(state, statuses, {})
        return statuses

    require_clean_git_snapshot(REPO_ROOT, expected_commit=expected_git_commit)

    by_id = {str(job["job_id"]): job for job in jobs}
    running: dict[str, tuple[subprocess.Popen, object, str, Path]] = {}
    runtime_terminal: dict[str, JobArtifactStatus] = {}
    failed = False
    while True:
        statuses = build_execution_plan(jobs)
        statuses.update(runtime_terminal)
        _apply_dependency_states(jobs, statuses)
        for job_id in running:
            statuses[job_id] = JobArtifactStatus("running")
        free_gpus = [gpu for gpu in gpus[:parallel] if gpu not in {row[2] for row in running.values()}]
        ready = [job_id for job_id, status in statuses.items() if status.status == "ready"]
        while not failed and ready and free_gpus:
            job_id = ready.pop(0)
            job = by_id[job_id]
            gpu = free_gpus.pop(0)
            config_path = Path(str(job["_config_path"]))
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "run_pipeline.py"),
                "--config",
                str(config_path),
                "--stages",
                str(job["stages"]),
            ]
            log_path = logs / f"{job_id.replace('.', '__')}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            handle.write("[RUN] " + " ".join(cmd) + f"\n[CUDA_VISIBLE_DEVICES] {gpu}\n")
            handle.flush()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[job_id] = (process, handle, gpu, log_path)
            statuses[job_id] = JobArtifactStatus("running")

        _write_state(state, statuses, running)
        if not running:
            remaining = [value.status for value in statuses.values() if value.status != "completed"]
            if not remaining:
                return statuses
            if failed or not any(value == "ready" for value in remaining):
                raise RuntimeError(f"G6 execution stopped with summary {summarize_execution_plan(statuses)}")

        time.sleep(max(float(poll_seconds), 0.1))
        for job_id, (process, handle, _gpu, log_path) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            running.pop(job_id)
            if return_code != 0:
                runtime_terminal[job_id] = JobArtifactStatus(
                    "failed", f"exit code {return_code}; log={log_path}"
                )
                failed = True
                continue
            verified = artifact_status(by_id[job_id])
            if verified.status != "completed":
                runtime_terminal[job_id] = JobArtifactStatus(
                    "failed", f"process exited 0 but artifact verification returned {verified}"
                )
                failed = True
            else:
                runtime_terminal.pop(job_id, None)
