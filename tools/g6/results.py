"""Validate G6 run records and aggregate the frozen result tables."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .matrix import ExperimentCell, build_required_cells


def load_run_records(root: str | Path) -> list[dict[str, Any]]:
    record_root = Path(root).expanduser().resolve()
    if not record_root.is_dir():
        raise FileNotFoundError(f"Run-record root not found: {record_root}")
    paths = sorted(record_root.rglob("run_record.json"))
    if not paths:
        raise ValueError(f"No run_record.json files found under {record_root}")
    records = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Run record must be a JSON object: {path}")
        payload["_record_path"] = str(path)
        records.append(payload)
    return records


def _evaluation_cells() -> dict[str, ExperimentCell]:
    return {
        cell.job_id: cell
        for cell in build_required_cells()
        if cell.job_type == "evaluate"
    }


def validate_run_records(
    records: list[dict[str, Any]],
    *,
    protocol_hash: str | None = None,
    expected_git_commit: str | None = None,
    expected_data_manifest_hashes: dict[str, str] | None = None,
    verify_artifacts: bool = False,
) -> str:
    expected = _evaluation_cells()
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        job_id = str(record.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("Run record has no job_id.")
        if job_id in by_id:
            raise ValueError(f"Duplicate run record: {job_id}")
        by_id[job_id] = record

    missing = sorted(set(expected) - set(by_id))
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Evaluation matrix is incomplete: missing={len(missing)}, extra={len(extra)}; "
            f"first_missing={missing[:1]}, first_extra={extra[:1]}"
        )

    observed_hashes: set[str] = set()
    observed_commits: set[str] = set()
    data_hashes: dict[tuple[str, int | None], str] = {}
    for job_id, cell in expected.items():
        record = by_id[job_id]
        if record.get("status") != "completed":
            raise ValueError(f"Required run is not completed: {job_id}")
        observed_hash = str(record.get("protocol_hash", "")).strip()
        if not observed_hash:
            raise ValueError(f"Run record has no protocol_hash: {job_id}")
        observed_hashes.add(observed_hash)
        if int(record.get("seed", -1)) != cell.seed:
            raise ValueError(f"Run record seed mismatch: {job_id}")
        if str(record.get("condition", "")) != cell.condition:
            raise ValueError(f"Run record condition mismatch: {job_id}")
        if str(record.get("dataset", "")) != cell.dataset:
            raise ValueError(f"Run record dataset mismatch: {job_id}")
        source = record.get("source")
        if source != cell.source:
            raise ValueError(f"Run record source mismatch: {job_id}")
        if record.get("test_session") != cell.test_session:
            raise ValueError(f"Run record test_session mismatch: {job_id}")
        if str(record.get("train_job_id", "")) != str(cell.depends_on):
            raise ValueError(f"Run record train_job_id mismatch: {job_id}")
        for required_artifact in (
            "config_hash",
            "resolved_config",
            "data_manifest_hash",
            "checkpoint",
            "checkpoint_sha256",
            "raw_results",
            "raw_results_sha256",
            "git_commit",
        ):
            if not str(record.get(required_artifact, "")).strip():
                raise ValueError(f"Run record has no {required_artifact}: {job_id}")
        if bool(record.get("git_dirty", True)):
            raise ValueError(f"Formal run record was produced from a dirty worktree: {job_id}")
        runtime = record.get("runtime")
        if not isinstance(runtime, dict):
            raise TypeError(f"Run record has no runtime object: {job_id}")
        for runtime_field in ("python", "torch", "cuda_build", "cuda_available"):
            if runtime_field not in runtime:
                raise ValueError(f"Run record runtime has no {runtime_field}: {job_id}")
        commit = str(record["git_commit"]).strip()
        observed_commits.add(commit)
        if expected_git_commit is not None and commit != expected_git_commit:
            raise ValueError(f"Run record Git commit mismatch: {job_id}")
        resolved_config = str(record["resolved_config"])
        if hashlib.sha256(resolved_config.encode("utf-8")).hexdigest() != str(record["config_hash"]):
            raise ValueError(f"Run record config hash mismatch: {job_id}")
        embedded = yaml.safe_load(resolved_config)
        if not isinstance(embedded, dict):
            raise TypeError(f"Run record resolved config is not a mapping: {job_id}")
        experiment = embedded.get("EXPERIMENT", {})
        embedded_checks = {
            "JOB_ID": job_id,
            "TRAIN_JOB_ID": str(cell.depends_on),
            "PROTOCOL_HASH": observed_hash,
            "DATA_MANIFEST_HASH": str(record["data_manifest_hash"]),
        }
        for field, expected_value in embedded_checks.items():
            if str(experiment.get(field, "")) != expected_value:
                raise ValueError(f"Run record resolved config {field} mismatch: {job_id}")
        if str(embedded.get("PREPROCESS", {}).get("DATASET", "")) != cell.dataset:
            raise ValueError(f"Run record resolved config dataset mismatch: {job_id}")
        if int(embedded.get("TRAIN", {}).get("SEED", -1)) != cell.seed:
            raise ValueError(f"Run record resolved config seed mismatch: {job_id}")
        for hash_field in ("checkpoint_sha256", "raw_results_sha256"):
            digest = str(record[hash_field]).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Run record has invalid {hash_field}: {job_id}")
        manifest_key = (cell.dataset, cell.fold_id if cell.dataset == "custom" else None)
        data_hash = str(record["data_manifest_hash"])
        previous_hash = data_hashes.setdefault(manifest_key, data_hash)
        if previous_hash != data_hash:
            raise ValueError(f"Run records mix data manifests for {manifest_key}: {job_id}")
        if verify_artifacts:
            for path_field, hash_field in (
                ("checkpoint", "checkpoint_sha256"),
                ("raw_results", "raw_results_sha256"),
            ):
                path = Path(str(record[path_field])).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Run record artifact is missing for {job_id}: {path}")
                if _sha256_file(path) != str(record[hash_field]):
                    raise ValueError(f"Run record artifact hash mismatch for {job_id}: {path_field}")
            raw_payload = json.loads(Path(str(record["raw_results"])).read_text(encoding="utf-8"))
            raw_frame = raw_payload.get("evaluations", {}).get("frame_acc", {})
            if raw_frame.get("prediction_schema_version") != "1.0":
                raise ValueError(f"Raw FrameAcc predictions have no supported schema: {job_id}")
            if raw_frame.get("mode") == "session_segments":
                clips = raw_frame.get("clips")
                if not isinstance(clips, list) or not clips:
                    raise ValueError(f"Raw segment FrameAcc has no clips: {job_id}")
                if any("frame_assignments" not in clip or "window_predictions" not in clip for clip in clips):
                    raise ValueError(f"Raw segment FrameAcc has incomplete assignments: {job_id}")
            elif not isinstance(raw_frame.get("assignments"), list) or not raw_frame["assignments"]:
                raise ValueError(f"Raw window FrameAcc has no assignments: {job_id}")

        metric = record.get("frame_acc")
        if not isinstance(metric, dict):
            raise TypeError(f"Run record has no frame_acc object: {job_id}")
        correct = int(metric.get("correct", -1))
        total = int(metric.get("total", 0))
        value = float(metric.get("value", -1.0))
        if total <= 0 or correct < 0 or correct > total:
            raise ValueError(f"Invalid FrameAcc counts for {job_id}: {correct}/{total}")
        expected_value = correct / total
        if abs(value - expected_value) > 1e-12:
            raise ValueError(
                f"FrameAcc value does not equal correct/total for {job_id}: "
                f"{value} != {correct}/{total}"
            )
        if cell.condition == "source":
            candidate_groups = record.get("candidate_groups")
            if not isinstance(candidate_groups, dict):
                raise TypeError(f"Source run record has no candidate_groups object: {job_id}")
            if float(candidate_groups.get("singleton_rate", -1.0)) != 0.0:
                raise ValueError(f"Source FrameAcc has nonzero singleton rate: {job_id}")
            if int(candidate_groups.get("candidate_group_size_min", 0)) < 2:
                raise ValueError(f"Source FrameAcc has a non-discriminative candidate group: {job_id}")

    if len(observed_hashes) != 1:
        raise ValueError(f"Run records mix protocol hashes: {sorted(observed_hashes)}")
    observed_hash = next(iter(observed_hashes))
    if protocol_hash is not None and observed_hash != protocol_hash:
        raise ValueError(
            f"Run-record protocol hash {observed_hash} does not match expected {protocol_hash}."
        )
    if len(observed_commits) != 1:
        raise ValueError(f"Run records mix Git commits: {sorted(observed_commits)}")
    if expected_data_manifest_hashes is not None:
        observed_data_hashes = {
            f"{dataset}.fold{fold_id if fold_id is not None else 'none'}": value
            for (dataset, fold_id), value in data_hashes.items()
        }
        if observed_data_hashes != expected_data_manifest_hashes:
            raise ValueError(
                "Run-record data manifests do not match the protocol record: "
                f"observed={observed_data_hashes}, expected={expected_data_manifest_hashes}"
            )
    return observed_hash


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_summary(values: dict[int, float]) -> dict[str, Any]:
    ordered = [float(values[seed]) for seed in sorted(values)]
    return {
        "by_seed": {str(seed): float(values[seed]) for seed in sorted(values)},
        "mean": float(statistics.mean(ordered)),
        "sample_std": float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def _bootstrap_mean_ci(values: list[float], *, seed: int = 20260812, trials: int = 10_000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty paired difference set.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(trials, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "trials": int(trials), "seed": int(seed)}


def _paired_comparison(
    left: dict[tuple[int, str], float],
    right: dict[tuple[int, str], float],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError(
            f"Paired comparison axes differ for {left_label} vs {right_label}: "
            f"left_only={sorted(set(left) - set(right))}, right_only={sorted(set(right) - set(left))}"
        )
    keys = sorted(left)
    differences = [float(left[key] - right[key]) for key in keys]
    difference_std = float(statistics.stdev(differences)) if len(differences) > 1 else 0.0
    mean_difference = float(statistics.mean(differences))
    return {
        "left": left_label,
        "right": right_label,
        "n_pairs": len(keys),
        "mean_paired_difference": mean_difference,
        "sample_std_difference": difference_std,
        "cohen_dz": float(mean_difference / difference_std) if difference_std > 0 else None,
        "bootstrap_mean_95ci": _bootstrap_mean_ci(differences),
        "differences_by_seed_session": {
            f"seed{seed}.{session}": float(left[(seed, session)] - right[(seed, session)])
            for seed, session in keys
        },
        "inference": "exploratory",
    }


def aggregate_run_records(
    records: list[dict[str, Any]],
    *,
    protocol_hash: str | None = None,
    expected_git_commit: str | None = None,
    expected_data_manifest_hashes: dict[str, str] | None = None,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    validated_hash = validate_run_records(
        records,
        protocol_hash=protocol_hash,
        expected_git_commit=expected_git_commit,
        expected_data_manifest_hashes=expected_data_manifest_hashes,
        verify_artifacts=verify_artifacts,
    )
    source_values: dict[str, dict[int, float]] = defaultdict(dict)
    session_records: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)

    for record in records:
        metric = record["frame_acc"]
        condition = str(record["condition"])
        seed = int(record["seed"])
        source = str(record.get("source") or "none")
        if condition == "source":
            source_values[source][seed] = float(metric["value"])
            continue
        session = str(record["test_session"])
        session_records[(condition, source, session)][seed] = record

    source_results = {
        source: _seed_summary(values)
        for source, values in sorted(source_values.items())
    }

    custom_sessions: dict[str, Any] = {}
    for (condition, source, session), by_seed in sorted(session_records.items()):
        key = f"{condition}.{source}.{session}"
        values = {
            seed: float(record["frame_acc"]["value"])
            for seed, record in by_seed.items()
        }
        custom_sessions[key] = {
            "condition": condition,
            "source": None if source == "none" else source,
            "session": session,
            **_seed_summary(values),
            "counts_by_seed": {
                str(seed): {
                    "correct": int(record["frame_acc"]["correct"]),
                    "total": int(record["frame_acc"]["total"]),
                }
                for seed, record in sorted(by_seed.items())
            },
        }

    overall_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["condition"] != "source":
            overall_groups[
                (str(record["condition"]), str(record.get("source") or "none"))
            ].append(record)

    custom_overall: dict[str, Any] = {}
    for (condition, source), group_records in sorted(overall_groups.items()):
        macro_by_seed: dict[int, float] = {}
        micro_by_seed: dict[int, float] = {}
        for seed in sorted({int(record["seed"]) for record in group_records}):
            seed_records = [record for record in group_records if int(record["seed"]) == seed]
            macro_by_seed[seed] = float(
                statistics.mean(float(record["frame_acc"]["value"]) for record in seed_records)
            )
            correct = sum(int(record["frame_acc"]["correct"]) for record in seed_records)
            total = sum(int(record["frame_acc"]["total"]) for record in seed_records)
            micro_by_seed[seed] = float(correct / total)
        key = f"{condition}.{source}"
        custom_overall[key] = {
            "condition": condition,
            "source": None if source == "none" else source,
            "macro_session": _seed_summary(macro_by_seed),
            "micro_weighted": _seed_summary(micro_by_seed),
            "session_sample_std": _seed_summary({
                seed: float(statistics.stdev(
                    float(record["frame_acc"]["value"])
                    for record in group_records
                    if int(record["seed"]) == seed
                ))
                for seed in macro_by_seed
            }),
        }

    values_by_condition: dict[tuple[str, str], dict[tuple[int, str], float]] = defaultdict(dict)
    for record in records:
        if record["condition"] == "source":
            continue
        condition = str(record["condition"])
        source = str(record.get("source") or "none")
        pair_key = (int(record["seed"]), str(record["test_session"]))
        values_by_condition[(condition, source)][pair_key] = float(record["frame_acc"]["value"])

    comparisons: dict[str, Any] = {}
    for source in ("totalcapture", "egohumans"):
        comparisons[f"finetune_vs_zero_shot.{source}"] = _paired_comparison(
            values_by_condition[("finetune", source)],
            values_by_condition[("zero_shot", source)],
            left_label=f"finetune.{source}",
            right_label=f"zero_shot.{source}",
        )
        comparisons[f"finetune_vs_direct.{source}"] = _paired_comparison(
            values_by_condition[("finetune", source)],
            values_by_condition[("direct", "none")],
            left_label=f"finetune.{source}",
            right_label="direct.none",
        )

    data_manifest_hashes: dict[str, str] = {}
    for record in records:
        cell = _evaluation_cells()[str(record["job_id"])]
        manifest_key = f"{cell.dataset}.fold{cell.fold_id if cell.dataset == 'custom' else 'none'}"
        data_manifest_hashes[manifest_key] = str(record["data_manifest_hash"])
    observed_git_commit = str(records[0]["git_commit"])
    runtime_variants = {
        json.dumps(record["runtime"], sort_keys=True)
        for record in records
    }

    return {
        "schema_version": "1.1",
        "protocol_hash": validated_hash,
        "git_commit": observed_git_commit,
        "data_manifest_hashes": dict(sorted(data_manifest_hashes.items())),
        "runtime_variants": [json.loads(value) for value in sorted(runtime_variants)],
        "num_evaluations": len(records),
        "source": source_results,
        "custom_by_session": custom_sessions,
        "custom_overall": custom_overall,
        "paired_comparisons": comparisons,
    }
