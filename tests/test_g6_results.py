from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
import yaml

from tools.g6.matrix import build_required_cells
from tools.g6.report import render_results_markdown
from tools.g6.results import aggregate_run_records, validate_run_records


def _complete_records():
    records = []
    for cell in build_required_cells():
        if cell.job_type != "evaluate":
            continue
        manifest_key = f"{cell.dataset}.fold{cell.fold_id if cell.dataset == 'custom' else 'none'}"
        data_manifest_hash = f"data-{manifest_key}"
        resolved_config = yaml.safe_dump(
            {
                "EXPERIMENT": {
                    "JOB_ID": cell.job_id,
                    "TRAIN_JOB_ID": cell.depends_on,
                    "PROTOCOL_HASH": "protocol-test-hash",
                    "DATA_MANIFEST_HASH": data_manifest_hash,
                },
                "PREPROCESS": {"DATASET": cell.dataset},
                "TRAIN": {"SEED": cell.seed},
            },
            sort_keys=True,
        )
        records.append(
            {
                "job_id": cell.job_id,
                "status": "completed",
                "protocol_hash": "protocol-test-hash",
                "train_job_id": cell.depends_on,
                "config_hash": hashlib.sha256(resolved_config.encode()).hexdigest(),
                "resolved_config": resolved_config,
                "data_manifest_hash": data_manifest_hash,
                "checkpoint": f"/checkpoint/{cell.job_id}.pt",
                "checkpoint_sha256": "a" * 64,
                "raw_results": f"/results/{cell.job_id}.json",
                "raw_results_sha256": "b" * 64,
                "git_commit": "c" * 40,
                "git_dirty": False,
                "runtime": {
                    "python": "3.10.0",
                    "torch": "2.1.0+cu118",
                    "cuda_build": "11.8",
                    "cuda_available": True,
                    "cuda_visible_devices": "0",
                },
                "condition": cell.condition,
                "dataset": cell.dataset,
                "source": cell.source,
                "seed": cell.seed,
                "test_session": cell.test_session,
                "frame_acc": {"correct": 5, "total": 10, "value": 0.5},
                "candidate_groups": (
                    {"singleton_rate": 0.0, "candidate_group_size_min": 2}
                    if cell.condition == "source"
                    else {}
                ),
            }
        )
    return records


def test_g6_results_require_complete_matrix_and_aggregate_all_axes():
    result = aggregate_run_records(
        _complete_records(),
        protocol_hash="protocol-test-hash",
    )

    assert result["num_evaluations"] == 66
    assert result["git_commit"] == "c" * 40
    assert len(result["data_manifest_hashes"]) == 6
    assert result["source"]["totalcapture"]["by_seed"] == {
        "0": 0.5,
        "42": 0.5,
        "123": 0.5,
    }
    assert len(result["custom_by_session"]) == 20
    assert len(result["custom_overall"]) == 5
    assert result["custom_overall"]["direct.none"]["macro_session"]["mean"] == 0.5
    assert result["custom_overall"]["direct.none"]["micro_weighted"]["mean"] == 0.5
    assert result["custom_overall"]["direct.none"]["session_sample_std"]["mean"] == 0.0
    assert len(result["paired_comparisons"]) == 4
    assert result["paired_comparisons"]["finetune_vs_direct.totalcapture"]["n_pairs"] == 12

    report = render_results_markdown(result)
    assert report.count("20260211_") == 20
    assert "TBD" not in report
    assert "each session" not in report
    assert "Macro-session" in report


def test_g6_results_compute_paired_differences_from_raw_counts():
    records = _complete_records()
    correct_by_condition = {
        ("zero_shot", "totalcapture"): 40,
        ("zero_shot", "egohumans"): 30,
        ("finetune", "totalcapture"): 60,
        ("finetune", "egohumans"): 55,
        ("direct", None): 50,
    }
    for record in records:
        key = (record["condition"], record["source"])
        if key in correct_by_condition:
            correct = correct_by_condition[key]
            record["frame_acc"] = {"correct": correct, "total": 100, "value": correct / 100}

    result = aggregate_run_records(records, protocol_hash="protocol-test-hash")
    comparison = result["paired_comparisons"]["finetune_vs_zero_shot.totalcapture"]
    assert comparison["mean_paired_difference"] == pytest.approx(0.2)
    assert comparison["bootstrap_mean_95ci"]["low"] == pytest.approx(0.2)
    assert comparison["bootstrap_mean_95ci"]["high"] == pytest.approx(0.2)
    assert comparison["cohen_dz"] is None


def test_g6_results_reverify_protocol_manifests_and_raw_prediction_artifacts(tmp_path):
    records = _complete_records()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    source_raw = tmp_path / "source-results.json"
    source_raw.write_text(
        '{"evaluations":{"frame_acc":{"prediction_schema_version":"1.0",'
        '"assignments":[{"status":"evaluated"}]}}}',
        encoding="utf-8",
    )
    custom_raw = tmp_path / "custom-results.json"
    custom_raw.write_text(
        '{"evaluations":{"frame_acc":{"prediction_schema_version":"1.0",'
        '"mode":"session_segments","clips":[{"frame_assignments":[],"window_predictions":[]}]}}}',
        encoding="utf-8",
    )
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(source_raw.read_bytes()).hexdigest()
    custom_hash = hashlib.sha256(custom_raw.read_bytes()).hexdigest()
    for record in records:
        record["checkpoint"] = str(checkpoint)
        record["checkpoint_sha256"] = checkpoint_hash
        raw = source_raw if record["condition"] == "source" else custom_raw
        record["raw_results"] = str(raw)
        record["raw_results_sha256"] = source_hash if raw == source_raw else custom_hash

    expected_manifests = {
        f"{cell.dataset}.fold{cell.fold_id if cell.dataset == 'custom' else 'none'}": (
            f"data-{cell.dataset}.fold{cell.fold_id if cell.dataset == 'custom' else 'none'}"
        )
        for cell in build_required_cells()
        if cell.job_type == "evaluate"
    }
    result = aggregate_run_records(
        records,
        protocol_hash="protocol-test-hash",
        expected_git_commit="c" * 40,
        expected_data_manifest_hashes=expected_manifests,
        verify_artifacts=True,
    )
    assert result["num_evaluations"] == 66

    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        aggregate_run_records(records, verify_artifacts=True)


def test_g6_results_reject_missing_duplicate_hash_and_count_errors():
    records = _complete_records()
    with pytest.raises(ValueError, match="incomplete"):
        validate_run_records(records[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_run_records(records + [deepcopy(records[0])])

    mixed_hash = deepcopy(records)
    mixed_hash[0]["protocol_hash"] = "different"
    mixed_config = yaml.safe_load(mixed_hash[0]["resolved_config"])
    mixed_config["EXPERIMENT"]["PROTOCOL_HASH"] = "different"
    mixed_hash[0]["resolved_config"] = yaml.safe_dump(mixed_config, sort_keys=True)
    mixed_hash[0]["config_hash"] = hashlib.sha256(
        mixed_hash[0]["resolved_config"].encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="mix protocol hashes"):
        validate_run_records(mixed_hash)

    wrong_value = deepcopy(records)
    wrong_value[0]["frame_acc"]["value"] = 0.6
    with pytest.raises(ValueError, match="does not equal correct/total"):
        validate_run_records(wrong_value)

    dirty = deepcopy(records)
    dirty[0]["git_dirty"] = True
    with pytest.raises(ValueError, match="dirty worktree"):
        validate_run_records(dirty)

    wrong_dependency = deepcopy(records)
    wrong_dependency[0]["train_job_id"] = "wrong"
    with pytest.raises(ValueError, match="train_job_id mismatch"):
        validate_run_records(wrong_dependency)
