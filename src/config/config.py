"""YACS config loader with legacy YAML compatibility."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from yacs.config import CfgNode as CN

from .defaults import get_cfg_defaults


def substitute_variables(obj: Any, variables: Dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: substitute_variables(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_variables(item, variables) for item in obj]
    if isinstance(obj, str):
        for key, value in variables.items():
            obj = obj.replace(f"${{{key}}}", value)
        return obj
    return obj


def _read_yaml(config_path: Union[str, Path], extra_variables: Dict[str, str] | None = None) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {path}")

    variables: Dict[str, str] = {}
    for key, value in data.items():
        if key.endswith("_root") and isinstance(value, str):
            variables[key] = value
    root_dir = data.get("root_dir")
    if root_dir and isinstance(root_dir, str):
        variables["root_dir"] = root_dir
    if extra_variables:
        variables.update(extra_variables)
    return substitute_variables(data, variables) if variables else data


def _upper_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k).upper(): _upper_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_upper_keys(v) for v in obj]
    return obj


def _lower_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k).lower(): _lower_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_lower_keys(v) for v in obj]
    return obj


def _normalize_legacy(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(raw)

    if "model" in data:
        data.setdefault("train", {})
        data["train"].setdefault("model", data.pop("model"))
    if "output" in data:
        data.setdefault("train", {})
        data["train"].setdefault("output", data.pop("output"))
    if "grouped_test" in data:
        data.setdefault("test", {})
        data["test"].setdefault("grouped_test", data.pop("grouped_test"))

    extract = data.get("extract")
    if isinstance(extract, dict):
        merge_keys = [k for k in extract if k.startswith("merge_") and k != "merge_tracklets"]
        if merge_keys:
            merge = extract.setdefault("merge_tracklets", {})
            for key in merge_keys:
                merge.setdefault(key[len("merge_") :], extract.pop(key))

    test = data.get("test")
    if isinstance(test, dict) and isinstance(test.get("grouped_test"), dict):
        metrics = test.setdefault("metrics", {})
        metrics.setdefault("group_test", test["grouped_test"])
        test.pop("grouped_test", None)

    if isinstance(test, dict):
        # Deprecated test entrypoints are intentionally ignored. Official
        # evaluation is limited to FrameAcc and Group Test.
        for key in ("matcher", "synchronous_test", "eval_mode", "mode", "chunk_windows", "window_size", "stride"):
            test.pop(key, None)

    train = data.get("train") if isinstance(data.get("train"), dict) else data.get("TRAIN")
    if isinstance(train, dict) and isinstance(train.get("model"), dict):
        _normalize_hybrid_model_cfg(train["model"])
    if isinstance(train, dict) and isinstance(train.get("MODEL"), dict):
        _normalize_hybrid_model_cfg(train["MODEL"])

    slice_cfg = data.get("slice") if isinstance(data.get("slice"), dict) else data.get("SLICE")
    if isinstance(slice_cfg, dict):
        for key in (
            "train_subjects",
            "val_subjects",
            "test_subjects",
            "train_sessions",
            "val_sessions",
            "test_sessions",
        ):
            if key in slice_cfg:
                slice_cfg[key] = _as_string_tuple(slice_cfg[key])
            upper_key = key.upper()
            if upper_key in slice_cfg:
                slice_cfg[upper_key] = _as_string_tuple(slice_cfg[upper_key])

    return data


def _normalize_hybrid_model_cfg(model: Dict[str, Any]) -> None:
    """Drop obsolete encoder keys so legacy YAML can load into hybrid-only defaults."""
    legacy_keys = {
        "motionbert_root",
        "motionbert_config",
        "motionbert_ckpt",
        "skip_motionbert_ckpt",
        "imu_ckpt",
        "embed_dim",
        "imu_encoder_type",
        "adapter_type",
        "physics_d_model",
        "physics_n_heads",
        "physics_num_layers",
        "physics_fs_hz",
        "physics_n_fft",
        "physics_dropout",
        "use_global_motion",
        "global_motion_input_dim",
        "global_motion_hidden_dim",
        "global_motion_num_layers",
        "global_motion_dropout",
        "global_motion_input_type",
        "global_motion_fusion_type",
        "global_motion_fusion_proj",
        "global_motion_root_source",
        "global_motion_train_only",
        "global_motion_aux_weight",
    }
    for key in list(model.keys()):
        if str(key).lower() in legacy_keys:
            model.pop(key, None)


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return (str(value).strip(),)


def _merge_present_sections(cfg: CN, raw_upper: Dict[str, Any]) -> None:
    """Merge only sections explicitly present in YAML.

    This preserves optional stage semantics. For example, missing EXTRACT means
    ExtractStage can still skip instead of being created by defaults.
    """
    always = {"PROJECT", "WORK_DIR", "ROOT_DIR", "PATHS"}
    for key, value in raw_upper.items():
        if key in always or key in cfg:
            cfg.merge_from_other_cfg(CN({key: value}))
        else:
            cfg.set_new_allowed(True)
            cfg.merge_from_other_cfg(CN({key: value}))
            cfg.set_new_allowed(False)


def _data_home(cfg: CN) -> Path:
    env_root = os.environ.get("REID_DATA_HOME", "").strip()
    configured = str(cfg.PATHS.DATA_HOME or "").strip()
    return Path(env_root or configured or "/data/fzliang/reid-project").expanduser().resolve()


def _dataset_dir_name(dataset: str) -> str:
    name = str(dataset or "dataset").strip().lower().replace("+", "_plus")
    if name.startswith("totalcapture"):
        return "totalcapture"
    if name.startswith("egohumans"):
        return "egohumans"
    if name.startswith("custom_plus"):
        return "custom_plus"
    if name.startswith("custom"):
        return "custom"
    return name or "dataset"


def _is_local_repo_data_path(value: str) -> bool:
    text = value.strip()
    return (
        text.startswith("./data/")
        or text.startswith("data/")
        or text.startswith("./artifacts")
        or text == "artifacts"
        or text.startswith("artifacts/")
    )


def _resolve_project_data_path(value: str, data_home: Path) -> str:
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    text = value.strip()
    normalized = text[2:] if text.startswith("./") else text
    if normalized.startswith("data/interim/"):
        return str(data_home / "interim" / normalized[len("data/interim/") :])
    if normalized == "data/interim":
        return str(data_home / "interim")
    if normalized.startswith("data/processed/"):
        return str(data_home / "processed" / normalized[len("data/processed/") :])
    if normalized == "data/processed":
        return str(data_home / "processed")
    if normalized.startswith("data/cache/"):
        return str(data_home / "cache" / normalized[len("data/cache/") :])
    if normalized == "data/cache":
        return str(data_home / "cache")
    if normalized.startswith("artifacts/"):
        return str(data_home / "artifacts" / normalized[len("artifacts/") :])
    if normalized == "artifacts":
        return str(data_home / "artifacts")
    if _is_local_repo_data_path(text):
        return str(data_home / normalized)
    return str(path.resolve())


def _normalize_known_output_paths(cfg: CN, data_home: Path) -> None:
    fields = [
        ("PREPROCESS", "RAW_ROOT"),
        ("PREPROCESS", "EXTRACTED_ROOT"),
        ("PREPROCESS", "POSE2D_OUTPUT_ROOT"),
        ("PREPROCESS", "OUTPUT"),
        ("EXTRACT", "MANIFEST_CSV"),
        ("EXTRACT", "RESULTS_ROOT"),
        ("EXTRACT", "COPY_FROM"),
        ("SLICE", "ROOT"),
        ("SLICE", "OUT_DIR"),
        ("SLICE", "SKELETON_ROOT"),
        ("SLICE", "SEGMENT_ROOT"),
        ("SLICE", "OUTPUT_ROOT"),
        ("SLICE", "CUSTOM_IMU_ROOT"),
        ("SLICE", "RAW_IMU_ROOT"),
        ("PATHS", "DATA_ROOT"),
        ("PATHS", "TRAIN_CSV"),
        ("PATHS", "VAL_CSV"),
        ("PATHS", "TEST_CSV"),
    ]
    for section, key in fields:
        if section in cfg and key in cfg[section]:
            value = str(cfg[section][key] or "")
            if value and not Path(value).expanduser().is_absolute():
                cfg[section][key] = _resolve_project_data_path(value, data_home)
    if "TRAIN" in cfg and "OUTPUT" in cfg.TRAIN:
        value = str(cfg.TRAIN.OUTPUT.OUTPUT_ROOT or "")
        if value and not Path(value).expanduser().is_absolute():
            cfg.TRAIN.OUTPUT.OUTPUT_ROOT = _resolve_project_data_path(value, data_home)
    if "TEST" in cfg and "OUTPUT" in cfg.TEST:
        value = str(cfg.TEST.OUTPUT.OUTPUT_ROOT or "")
        if value and not Path(value).expanduser().is_absolute():
            cfg.TEST.OUTPUT.OUTPUT_ROOT = _resolve_project_data_path(value, data_home)


def _resolve_paths(cfg: CN) -> CN:
    cfg.defrost()
    project = cfg.PROJECT or "reid_project"
    data_home = _data_home(cfg)
    cfg.PATHS.DATA_HOME = str(data_home)
    dataset_name = _dataset_dir_name(cfg.PREPROCESS.DATASET)
    dataset_home = data_home / dataset_name
    preprocessed_dir = dataset_home / "preprocessed" / project
    skeleton_dir = dataset_home / "skeleton" / str(cfg.EXTRACT.POSE_ESTIMATOR or "alphapose").strip().lower()
    imu_source = str(cfg.PREPROCESS.IMU_SOURCE or "raw").strip().lower() or "raw"
    imu_dir = dataset_home / "imu" / imu_source
    artifacts_dir = dataset_home / "artifacts" / project

    work_dir = Path(cfg.WORK_DIR).expanduser() if cfg.WORK_DIR else preprocessed_dir
    if not work_dir.is_absolute():
        work_dir = Path(_resolve_project_data_path(str(work_dir), data_home))
    work_dir = work_dir.resolve()
    cfg.WORK_DIR = str(work_dir)

    if cfg.PREPROCESS.DATASET:
        if not cfg.PREPROCESS.get("OUTPUT", ""):
            cfg.PREPROCESS.OUTPUT = str(preprocessed_dir / "video_manifest.csv")
        if not cfg.PREPROCESS.POSE2D_OUTPUT_ROOT:
            cfg.PREPROCESS.POSE2D_OUTPUT_ROOT = str(skeleton_dir)

    if "EXTRACT" in cfg and cfg.EXTRACT.POSE_ESTIMATOR:
        if not cfg.EXTRACT.RESULTS_ROOT:
            cfg.EXTRACT.RESULTS_ROOT = str(skeleton_dir)
        if cfg.PREPROCESS.DATASET and not cfg.EXTRACT.MANIFEST_CSV:
            cfg.EXTRACT.MANIFEST_CSV = cfg.PREPROCESS.OUTPUT

    if "SLICE" in cfg and cfg.SLICE.WINDOW_LEN:
        if not cfg.SLICE.OUT_DIR:
            cfg.SLICE.OUT_DIR = str(preprocessed_dir)
        if cfg.PREPROCESS.DATASET and not cfg.SLICE.ROOT:
            cfg.SLICE.ROOT = str(preprocessed_dir)
        if cfg.SLICE.SKELETON_SOURCE == "alphapose" and cfg.EXTRACT.RESULTS_ROOT and not cfg.SLICE.SKELETON_ROOT:
            cfg.SLICE.SKELETON_ROOT = cfg.EXTRACT.RESULTS_ROOT

    if not cfg.PATHS.DATA_ROOT:
        if cfg.SLICE.OUT_DIR:
            cfg.PATHS.DATA_ROOT = cfg.SLICE.OUT_DIR
        elif cfg.SLICE.ROOT:
            cfg.PATHS.DATA_ROOT = cfg.SLICE.ROOT
    if cfg.PATHS.DATA_ROOT:
        root = Path(cfg.PATHS.DATA_ROOT)
        if not cfg.PATHS.TRAIN_CSV:
            cfg.PATHS.TRAIN_CSV = str(root / "windows_train.csv")
        if not cfg.PATHS.VAL_CSV:
            cfg.PATHS.VAL_CSV = str(root / "windows_val.csv")
        if not cfg.PATHS.TEST_CSV:
            cfg.PATHS.TEST_CSV = str(root / "windows_test.csv")

    if "TRAIN" in cfg:
        if not cfg.TRAIN.OUTPUT.OUTPUT_ROOT:
            cfg.TRAIN.OUTPUT.OUTPUT_ROOT = str(artifacts_dir / "train")
        if not cfg.TRAIN.OUTPUT.RUN_NAME:
            cfg.TRAIN.OUTPUT.RUN_NAME = project

    if "TEST" in cfg:
        if not cfg.TEST.OUTPUT.OUTPUT_ROOT:
            cfg.TEST.OUTPUT.OUTPUT_ROOT = str(artifacts_dir / "evaluate")
        if not cfg.TEST.OUTPUT.RUN_NAME:
            cfg.TEST.OUTPUT.RUN_NAME = project

    _normalize_known_output_paths(cfg, data_home)

    if "TRAIN" in cfg and str(cfg.TRAIN.MODEL.TYPE).lower() == "hybrid":
        # The hybrid encoder consumes raw 7D IMU. Legacy configs often set
        # IMU_SENSOR/R_LowArm to expand one sensor into the old 48D LSTM format;
        # that path is incompatible with the hybrid default.
        cfg.TRAIN.IMU_SENSOR = ""
        cfg.TRAIN.REPEAT_SINGLE_SENSOR = 1
    return cfg


def cfg_to_dict(cfg: CN) -> Dict[str, Any]:
    return _lower_keys(yaml.safe_load(cfg.dump(sort_keys=False)))


def _load_raw_normalized(config_path: Union[str, Path], extra_variables: Dict[str, str] | None = None) -> Dict[str, Any]:
    return _normalize_legacy(_read_yaml(config_path, extra_variables=extra_variables))


def load_cfg(config_path: Union[str, Path], extra_variables: Dict[str, str] | None = None) -> CN:
    raw = _load_raw_normalized(config_path, extra_variables=extra_variables)
    cfg = get_cfg_defaults()
    _merge_present_sections(cfg, _upper_keys(raw))
    _resolve_paths(cfg)
    cfg.freeze()
    return cfg


def load_config(config_path: Union[str, Path], extra_variables: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Load YAML, merge YACS defaults, resolve paths, and return legacy dict."""
    raw = _load_raw_normalized(config_path, extra_variables=extra_variables)
    data = cfg_to_dict(load_cfg(config_path, extra_variables=extra_variables))
    present_sections = {str(k).lower() for k in raw}

    # Keep the old pipeline contract: optional stage sections are absent unless
    # the YAML explicitly requested them. Paths remain resolved globally.
    for section in ("preprocess", "extract", "slice", "train", "test"):
        if section not in present_sections and section in data:
            data.pop(section)
    return data


def resolve_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    return load_config(config_path)
