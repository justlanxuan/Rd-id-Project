"""Configuration utilities with variable substitution and smart defaults."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def substitute_variables(obj: Any, variables: Dict[str, str]) -> Any:
    """Recursively substitute ${var} variables in config objects.

    Args:
        obj: Configuration object (dict, list, or primitive)
        variables: Mapping of variable name -> replacement value

    Returns:
        Configuration with substituted variables
    """
    if isinstance(obj, dict):
        return {k: substitute_variables(v, variables) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_variables(item, variables) for item in obj]
    elif isinstance(obj, str):
        for key, value in variables.items():
            obj = obj.replace(f"${{{key}}}", value)
        return obj
    else:
        return obj


def load_config(config_path: Union[str, Path], extra_variables: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Load YAML config file with variable substitution.

    Supports ${var} variables. By default, all top-level keys ending with
    `_root` are treated as variables. Additionally, `root_dir` is supported
    for backward compatibility. Optional extra_variables can be injected.

    Args:
        config_path: Path to YAML config file
        extra_variables: Optional extra variables to inject

    Returns:
        Configuration dictionary with substituted variables
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {path}")

    # Collect variables: all top-level *_root keys + legacy root_dir
    variables: Dict[str, str] = {}
    for key, value in data.items():
        if key.endswith("_root") and isinstance(value, str):
            variables[key] = value
    root_dir = data.get("root_dir")
    if root_dir and isinstance(root_dir, str):
        variables["root_dir"] = root_dir
    if extra_variables:
        variables.update(extra_variables)

    if variables:
        data = substitute_variables(data, variables)

    return data


def resolve_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and resolve a workflow config with auto-derived paths and hierarchical normalization.

    This function:
      1. Loads the raw YAML.
      2. Normalizes legacy flat keys (model, output, grouped_test, merge_*) into the new hierarchy.
      3. Generates a timestamped work_dir from the project name.
      4. Auto-derives stage I/O paths so outputs chain into inputs automatically.

    Args:
        config_path: Path to YAML workflow config.

    Returns:
        Fully resolved configuration dictionary.
    """
    cfg = load_config(config_path)

    # ------------------------------------------------------------------
    # 1. Normalize legacy flat layout into hierarchy
    # ------------------------------------------------------------------
    # model -> train.model
    if "model" in cfg:
        cfg.setdefault("train", {})
        if "model" not in cfg["train"]:
            cfg["train"]["model"] = cfg.pop("model")
        else:
            # Prefer existing train.model; drop stale top-level model
            cfg.pop("model")

    # output -> train.output
    if "output" in cfg:
        cfg.setdefault("train", {})
        if "output" not in cfg["train"]:
            cfg["train"]["output"] = cfg.pop("output")
        else:
            cfg.pop("output")

    # grouped_test -> test.grouped_test
    if "grouped_test" in cfg:
        cfg.setdefault("test", {})
        if "grouped_test" not in cfg["test"]:
            cfg["test"]["grouped_test"] = cfg.pop("grouped_test")
        else:
            cfg.pop("grouped_test")

    # extract.merge_* -> extract.merge_tracklets.*
    extract = cfg.get("extract")
    if isinstance(extract, dict):
        # Preserve nested merge_tracklets config; only migrate legacy flat merge_* keys.
        merge_keys = [k for k in extract if k.startswith("merge_") and k != "merge_tracklets"]
        if merge_keys:
            merge = extract.setdefault("merge_tracklets", {})
            for k in merge_keys:
                new_k = k[len("merge_") :]
                if new_k not in merge:
                    merge[new_k] = extract.pop(k)
                else:
                    extract.pop(k)

    # ------------------------------------------------------------------
    # 2. Generate work_dir (stable path under data/interim)
    # ------------------------------------------------------------------
    project = cfg.get("project", "autism_project")
    work_dir = Path(cfg.get("work_dir", f"./data/interim/{project}")).expanduser().resolve()
    cfg["work_dir"] = str(work_dir)

    # ------------------------------------------------------------------
    # 3. Auto-derive stage paths
    # ------------------------------------------------------------------
    preprocess = cfg.get("preprocess")
    if isinstance(preprocess, dict):
        preprocess.setdefault("output", str(work_dir / "preprocess" / "video_manifest.csv"))

    # Extract stage
    if isinstance(extract, dict):
        extract.setdefault("results_root", str(work_dir / "extract"))
        if isinstance(preprocess, dict):
            extract.setdefault("manifest_csv", preprocess.get("output"))

    # Slice stage
    slice_cfg = cfg.get("slice")
    if isinstance(slice_cfg, dict):
        slice_cfg.setdefault("out_dir", str(work_dir / "slice"))
        # Default slice root to preprocess output dir when preprocess is present
        if isinstance(preprocess, dict):
            slice_cfg.setdefault("root", str(work_dir / "preprocess"))

        # skeleton_source inherits from extract.pose_estimator, else defaults to vicon
        extract_present = isinstance(extract, dict)
        pose_estimator = extract.get("pose_estimator") if extract_present else None
        if pose_estimator:
            slice_cfg.setdefault("skeleton_source", pose_estimator)
        else:
            slice_cfg.setdefault("skeleton_source", "vicon")

        if slice_cfg.get("skeleton_source") == "alphapose" and extract_present:
            slice_cfg.setdefault("skeleton_root", extract.get("results_root"))

    # Paths (used by train/test)
    cfg.setdefault("paths", {})
    paths = cfg["paths"]
    if isinstance(slice_cfg, dict):
        paths.setdefault("data_root", slice_cfg["out_dir"])
    data_root = paths.get("data_root")
    if data_root:
        paths.setdefault("train_csv", str(Path(data_root) / "windows_train.csv"))
        paths.setdefault("val_csv", str(Path(data_root) / "windows_val.csv"))
        paths.setdefault("test_csv", str(Path(data_root) / "windows_test.csv"))

    # Train output
    train_cfg = cfg.get("train")
    if isinstance(train_cfg, dict):
        train_cfg.setdefault("output", {})
        output = train_cfg["output"]
        output.setdefault("output_root", str(work_dir / "train"))
        output.setdefault("run_name", project)

    # Test output
    test_cfg = cfg.get("test")
    if isinstance(test_cfg, dict):
        test_cfg.setdefault("output", {})
        output = test_cfg["output"]
        output.setdefault("output_root", str(work_dir / "test"))
        output.setdefault("run_name", project)

    return cfg
