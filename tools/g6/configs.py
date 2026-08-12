"""Generate one resolved workflow YAML for every G6 required cell."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from src.config import load_cfg

from .build_data_manifests import CUSTOM_ROOT, SOURCE_ROOTS
from .matrix import ExperimentCell, build_required_cells

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIGS = {
    "totalcapture": REPO_ROOT / "configs/g6/totalcapture_source.yaml",
    "egohumans": REPO_ROOT / "configs/g6/egohumans_source.yaml",
}
CUSTOM_BASE_CONFIGS = {
    fold_id: REPO_ROOT / f"configs/g6/custom_direct_fold{fold_id}.yaml"
    for fold_id in range(1, 5)
}


def _safe_name(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", job_id).replace(".", "__")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Base config must be a mapping: {path}")
    return payload


def _manifest_hashes(index_path: Path) -> dict[tuple[str, int | None], str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    output: dict[tuple[str, int | None], str] = {}
    for row in index.get("manifests", []):
        key = (str(row["dataset"]), row.get("fold_id"))
        manifest_hash = str(row.get("manifest_hash", "")).strip()
        if not manifest_hash:
            raise ValueError(f"Data manifest index has an empty hash: {row}")
        if key in output:
            raise ValueError(f"Duplicate data manifest index key: {key}")
        output[key] = manifest_hash
    expected = {("totalcapture", None), ("egohumans", None)} | {
        ("custom", fold_id) for fold_id in range(1, 5)
    }
    if set(output) != expected:
        raise ValueError(f"Data manifest index mismatch: expected={expected}, actual={set(output)}")
    return output


def _prepared_root(cell: ExperimentCell) -> Path:
    if cell.dataset in SOURCE_ROOTS:
        return SOURCE_ROOTS[cell.dataset]
    assert cell.fold_id is not None
    return CUSTOM_ROOT / f"fold{cell.fold_id}_{cell.test_session}"


def _base_config(cell: ExperimentCell) -> dict[str, Any]:
    if cell.dataset in BASE_CONFIGS:
        return _load_yaml(BASE_CONFIGS[cell.dataset])
    assert cell.fold_id is not None
    return _load_yaml(CUSTOM_BASE_CONFIGS[cell.fold_id])


def _checkpoint_for_train_job(train_job_id: str, artifact_root: Path) -> Path:
    return artifact_root / "train" / _safe_name(train_job_id) / "best.pt"


def _configure_data(config: dict[str, Any], cell: ExperimentCell) -> None:
    root = _prepared_root(cell)
    preprocess = config.setdefault("preprocess", {})
    preprocess["dataset"] = cell.dataset
    preprocess["reuse_prepared"] = True
    preprocess["prepared_root"] = str(root)
    config["paths"] = {
        "data_root": str(root),
        "train_csv": str(root / "windows_train.csv"),
        "val_csv": str(root / "windows_val.csv"),
        "test_csv": str(root / "windows_test.csv"),
    }


def _configure_train(
    config: dict[str, Any],
    cell: ExperimentCell,
    *,
    protocol_hash: str,
    data_manifest_hash: str,
    artifact_root: Path,
) -> None:
    train = config.setdefault("train", {})
    train["seed"] = cell.seed
    train.setdefault("model", {})
    train["model"]["init_alignment_ckpt"] = ""
    if cell.condition == "finetune":
        assert cell.depends_on is not None
        train["model"]["init_alignment_ckpt"] = str(
            _checkpoint_for_train_job(cell.depends_on, artifact_root)
        )
    train["output"] = {
        "output_root": str(artifact_root / "train"),
        "run_name": _safe_name(cell.job_id),
    }
    test = config.setdefault("test", {})
    test["checkpoint"] = str(_checkpoint_for_train_job(cell.job_id, artifact_root))
    test["output"] = {
        "output_root": str(artifact_root / "evaluate_diagnostics"),
        "run_name": _safe_name(cell.job_id),
    }
    config["experiment"] = {
        "job_id": "",
        "train_job_id": cell.job_id,
        "protocol_hash": protocol_hash,
        "condition": cell.condition,
        "source": cell.source or "",
        "test_session": cell.test_session or "",
        "data_manifest_hash": data_manifest_hash,
        "run_record": "",
    }


def _configure_evaluation(
    config: dict[str, Any],
    cell: ExperimentCell,
    *,
    protocol_hash: str,
    data_manifest_hash: str,
    artifact_root: Path,
) -> None:
    assert cell.depends_on is not None
    train = config.setdefault("train", {})
    train["seed"] = cell.seed
    train.setdefault("model", {})["init_alignment_ckpt"] = ""
    train["output"] = {
        "output_root": str(artifact_root / "train"),
        "run_name": _safe_name(cell.depends_on),
    }
    test = config.setdefault("test", {})
    test["checkpoint"] = str(_checkpoint_for_train_job(cell.depends_on, artifact_root))
    test["output"] = {
        "output_root": str(artifact_root / "evaluate"),
        "run_name": _safe_name(cell.job_id),
    }
    test.setdefault("metrics", {}).setdefault("group_test", {})["enabled"] = False
    test["metrics"].setdefault("frame_acc", {})["seed"] = cell.seed
    config["experiment"] = {
        "job_id": cell.job_id,
        "train_job_id": cell.depends_on,
        "protocol_hash": protocol_hash,
        "condition": cell.condition,
        "source": cell.source or "",
        "test_session": cell.test_session or "",
        "data_manifest_hash": data_manifest_hash,
        "run_record": str(artifact_root / "records" / _safe_name(cell.job_id) / "run_record.json"),
    }


def generate_resolved_configs(
    output_dir: str | Path,
    *,
    protocol_hash: str,
    data_manifest_index: str | Path,
    artifact_root: str | Path,
) -> list[dict[str, Any]]:
    """Write and validate 42 train plus 66 evaluation configs."""

    frozen_hash = str(protocol_hash).strip()
    if not frozen_hash:
        raise ValueError("protocol_hash is required to generate resolved formal configs.")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = Path(artifact_root).expanduser().resolve()
    manifest_hashes = _manifest_hashes(Path(data_manifest_index).expanduser().resolve())
    entries: list[dict[str, Any]] = []

    for cell in build_required_cells():
        config = copy.deepcopy(_base_config(cell))
        config["project"] = _safe_name(cell.job_id)
        _configure_data(config, cell)
        manifest_key = (cell.dataset, cell.fold_id if cell.dataset == "custom" else None)
        if cell.job_type == "train":
            _configure_train(
                config,
                cell,
                protocol_hash=frozen_hash,
                data_manifest_hash=manifest_hashes[manifest_key],
                artifact_root=artifacts,
            )
            stages = "preprocess,train"
        else:
            _configure_evaluation(
                config,
                cell,
                protocol_hash=frozen_hash,
                data_manifest_hash=manifest_hashes[manifest_key],
                artifact_root=artifacts,
            )
            stages = "test"

        filename = f"{_safe_name(cell.job_id)}.yaml"
        path = destination / filename
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_cfg(path)
        entries.append(
            {
                **cell.to_dict(),
                "config": filename,
                "stages": stages,
                "command": f"python run_pipeline.py --config {path} --stages {stages}",
            }
        )

    return entries
