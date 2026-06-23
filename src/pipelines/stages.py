"""Pipeline stage implementations."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from src.pipelines.base import PipelineStage
from src.utils.config import load_config, resolve_config


def _append_arg(cmd: list[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(key)
        return
    value_s = str(value).strip()
    if value_s:
        cmd.extend([key, value_s])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_with_pythonpath() -> dict:
    env = os.environ.copy()
    root = str(_repo_root())
    src = str(_repo_root() / "src")
    current = env.get("PYTHONPATH", "").strip()
    parts = [p for p in [root, src, current] if p]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else str(_repo_root()), env=_env_with_pythonpath())


def _run_dir_from_section(section: dict, default_root_name: str) -> Path:
    out = section.get("output", {}) if isinstance(section, dict) else {}
    return (
        _repo_root()
        / out.get("output_root", default_root_name)
        / out.get("run_name", "")
    ).resolve()


def _resolve_imu_stats_for_eval(test_cfg: dict, best_ckpt: Path | None) -> Path | None:
    """Resolve IMU stats json for evaluation.

    Priority:
      1) test.imu_stats_json (if provided and exists)
      2) sibling imu_stats.json next to checkpoint
    """
    override = ""
    if isinstance(test_cfg, dict):
        override = str(test_cfg.get("imu_stats_json", "") or "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        if p.exists():
            return p
        print(f"[WARN] test.imu_stats_json not found: {p}; falling back to checkpoint sibling.")

    if best_ckpt is not None:
        cand = best_ckpt.parent / "imu_stats.json"
        if cand.exists():
            return cand.resolve()
    return None


def _append_imu_filter_args(cmd: list[str], imu_cfg: dict) -> None:
    if not isinstance(imu_cfg, dict):
        return

    def _parse_float_or_none(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs.lower() in ("none", "null", ""):
                return None
            try:
                return float(xs)
            except ValueError:
                return None
        return None

    cutoff = _parse_float_or_none(imu_cfg.get("lowpass_cutoff_hz"))
    if cutoff is not None and cutoff > 0:
        cmd.extend(["--imu_lowpass_cutoff_hz", str(cutoff)])
    fs_hz = _parse_float_or_none(imu_cfg.get("lowpass_fs_hz"))
    if fs_hz is not None and fs_hz > 0:
        cmd.extend(["--imu_lowpass_fs_hz", str(fs_hz)])

class PreprocessStage(PipelineStage):
    """Run raw-data preprocessing according to dataset config."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = resolve_config(config_path)
        preprocess_cfg = cfg.get("preprocess", {})
        dataset = preprocess_cfg.get("dataset", "")

        if dataset in {"totalcapture", "totalcapture_synthetic"}:
            script = "-m"
            module = "src.data.preprocess.totalcapture"
        elif dataset == "custom":
            script = "-m"
            module = "src.data.preprocess.custom"
        else:
            print(f"[INFO] Preprocess stage not implemented for dataset '{dataset}'; skipping.")
            return state

        manifest_csv = preprocess_cfg.get("output")
        output_dir = str(Path(manifest_csv).parent) if manifest_csv else str(Path(cfg.get("work_dir", "")) / "preprocess")
        cmd = [
            sys.executable, script, module,
            "--config", str(config_path),
            "--output_dir", output_dir,
        ]
        if manifest_csv:
            cmd.extend(["--manifest_csv", manifest_csv])

        _run(cmd)
        work_dir = cfg.get("work_dir", "")
        state["preprocess_dir"] = str(Path(work_dir) / "preprocess")
        return state


class SliceStage(PipelineStage):
    """Run IMU-skeleton slicing according to dataset config."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = resolve_config(config_path)
        preprocess_cfg = cfg.get("preprocess", {})
        dataset = preprocess_cfg.get("dataset", "")

        script = "-m"
        module = "src.data.slice.totalcapture"
        cmd = [sys.executable, script, module, "--config", str(config_path)]

        _run(cmd)
        # Pass through the data_root for downstream stages
        slice_cfg = cfg.get("slice", {})
        out_dir = slice_cfg.get("out_dir")
        if out_dir:
            state["data_root"] = out_dir
        return state


class ExtractStage(PipelineStage):
    """Run video skeleton extraction if config contains an extract section."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = resolve_config(config_path)
        extract_cfg = cfg.get("extract")
        if not isinstance(extract_cfg, dict):
            print("[INFO] No extract section in config; skipping extraction stage.")
            return state

        copy_from = str(extract_cfg.get("copy_from", "") or "").strip()
        if copy_from:
            src = Path(copy_from).expanduser().resolve()
            dst_root = extract_cfg.get("results_root", "")
            if dst_root:
                dst = Path(dst_root).expanduser().resolve()
            else:
                dst = (_repo_root() / "data" / "interim" / "extract").resolve()
            if not src.exists():
                raise FileNotFoundError(f"extract.copy_from not found: {src}")
            if dst.exists() and extract_cfg.get("skip_existing", True):
                print(f"[INFO] Extract copy skipped (exists): {dst}")
                return state
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[INFO] Copied extract results: {src} -> {dst}")
            return state

        script_path = (_repo_root() / "src" / "pipelines" / "video_pipeline" / "dispatcher.py").resolve()
        cmd = [sys.executable, str(script_path), "--config", str(config_path)]

        _run(cmd)
        return state


class TrainStage(PipelineStage):
    """Run IMU-video alignment training."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = resolve_config(config_path)
        train_cfg = cfg.get("train", {})
        model = train_cfg.get("model", {})
        paths = cfg.get("paths", {})
        out = train_cfg.get("output", {})
        train = train_cfg
        folds = cfg.get("folds")

        if isinstance(folds, list) and folds:
            for fold in folds:
                fold_cfg = _expand_cfg(cfg, int(fold))
                print(f"[INFO] Running fold {fold}")
                self._run_train(config_path, fold_cfg)
        else:
            self._run_train(config_path, cfg)

        return state

    def _run_train(self, config_path: Path, cfg: dict) -> None:
        train_cfg = cfg.get("train", {})
        model = train_cfg.get("model", {})
        paths = cfg.get("paths", {})
        out = train_cfg.get("output", {})
        train = train_cfg

        cmd = [
            sys.executable,
            "-m",
            "src.engine.train",
            "--train_csv", str(paths.get("train_csv", "")),
            "--val_csv", str(paths.get("val_csv", "")),
            "--data_root", str(paths.get("data_root", "")),
            "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
            "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
            "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
            "--imu_ckpt", str(model.get("imu_ckpt", "")),
            "--epochs", str(train.get("epochs", 40)),
            "--batch_size", str(train.get("batch_size", 64)),
            "--num_workers", str(train.get("num_workers", 8)),
            "--output_root", str(out.get("output_root", "artifacts")),
            "--run_name", str(out.get("run_name", "")),
        ]

        if train.get("compute_imu_stats"):
            cmd.append("--compute_imu_stats")
        if train.get("shuffle_video_in_batch"):
            cmd.append("--shuffle_video_in_batch")
        if train.get("device"):
            cmd.extend(["--device", str(train["device"])])
        if "imu_sensor" in train and train.get("imu_sensor") is not None:
            cmd.extend(["--imu_sensor", str(train.get("imu_sensor", ""))])
        if train.get("repeat_single_sensor") is not None:
            cmd.extend(["--repeat_single_sensor", str(train["repeat_single_sensor"])])
        if train.get("freeze_backbone_epochs") is not None:
            cmd.extend(["--freeze_backbone_epochs", str(train.get("freeze_backbone_epochs"))])
        if train.get("lr_backbone") is not None:
            cmd.extend(["--lr_backbone", str(train.get("lr_backbone"))])
        if train.get("lr_heads") is not None:
            cmd.extend(["--lr_heads", str(train.get("lr_heads"))])
        if train.get("weight_decay") is not None:
            cmd.extend(["--weight_decay", str(train.get("weight_decay"))])
        if train.get("adapter_train_only"):
            cmd.append("--adapter_train_only")
        _append_imu_filter_args(cmd, cfg.get("preprocess", {}).get("imu", {}))

        # Physics encoder args
        mdl = train.get("model", {})
        if mdl.get("imu_encoder_type"):
            cmd.extend(["--imu_encoder_type", str(mdl["imu_encoder_type"])])
        if mdl.get("physics_d_model") is not None:
            cmd.extend(["--physics_d_model", str(mdl["physics_d_model"])])
        if mdl.get("physics_n_heads") is not None:
            cmd.extend(["--physics_n_heads", str(mdl["physics_n_heads"])])
        if mdl.get("physics_num_layers") is not None:
            cmd.extend(["--physics_num_layers", str(mdl["physics_num_layers"])])
        if mdl.get("physics_fs_hz") is not None:
            cmd.extend(["--physics_fs_hz", str(mdl["physics_fs_hz"])])
        if mdl.get("physics_n_fft") is not None:
            cmd.extend(["--physics_n_fft", str(mdl["physics_n_fft"])])
        if mdl.get("physics_dropout") is not None:
            cmd.extend(["--physics_dropout", str(mdl["physics_dropout"])])

        # Global motion encoder args
        if mdl.get("use_global_motion"):
            cmd.append("--use_global_motion")
        if mdl.get("global_motion_input_dim") is not None:
            cmd.extend(["--global_motion_input_dim", str(mdl["global_motion_input_dim"])])
        if mdl.get("global_motion_hidden_dim") is not None:
            cmd.extend(["--global_motion_hidden_dim", str(mdl["global_motion_hidden_dim"])])
        if mdl.get("global_motion_num_layers") is not None:
            cmd.extend(["--global_motion_num_layers", str(mdl["global_motion_num_layers"])])
        if mdl.get("global_motion_dropout") is not None:
            cmd.extend(["--global_motion_dropout", str(mdl["global_motion_dropout"])])
        if mdl.get("global_motion_input_type"):
            cmd.extend(["--global_motion_input_type", str(mdl["global_motion_input_type"])])
        if mdl.get("global_motion_fusion_type"):
            cmd.extend(["--global_motion_fusion_type", str(mdl["global_motion_fusion_type"])])
        if mdl.get("global_motion_fusion_proj"):
            cmd.append("--global_motion_fusion_proj")
        if mdl.get("global_motion_root_source"):
            cmd.extend(["--global_motion_root_source", str(mdl["global_motion_root_source"])])
        if mdl.get("global_motion_aux_weight") is not None:
            cmd.extend(["--global_motion_aux_weight", str(mdl["global_motion_aux_weight"])])
        if mdl.get("global_motion_train_only"):
            cmd.append("--global_motion_train_only")
        if mdl.get("init_alignment_ckpt"):
            cmd.extend(["--init_alignment_ckpt", str(mdl["init_alignment_ckpt"])])

        _run(cmd)


class TestStage(PipelineStage):
    """Run standard evaluation and optional grouped / synchronous evaluation."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = resolve_config(config_path)
        train_cfg = cfg.get("train", {})
        test_cfg = cfg.get("test", {})
        model = train_cfg.get("model", {})
        paths = cfg.get("paths", {})
        test = test_cfg
        grouped = test.get("grouped_test", {})
        synchronous = test.get("synchronous_test", {})
        matcher_cfg = test.get("matcher", {})
        physics_cfg = matcher_cfg.get("physics_based_matcher", {})
        dl_cfg = matcher_cfg.get("dl_matcher", {})

        test_run_dir = _run_dir_from_section(test_cfg, "test")
        test_run_dir.mkdir(parents=True, exist_ok=True)

        physics_enabled = bool(physics_cfg.get("enabled", False))
        dl_enabled = bool(dl_cfg.get("enabled", True))
        eval_run_name = str(test.get("output", {}).get("run_name", "") or train_cfg.get("output", {}).get("run_name", "")).strip()

        best_ckpt = None
        imu_stats_for_eval = None
        if not physics_enabled:
            if not eval_run_name:
                raise ValueError("test.output.run_name is empty and no train.output.run_name fallback is available.")

            train_output_root = Path(str(train_cfg.get("output", {}).get("output_root", _repo_root() / "train"))).expanduser()
            train_run_dir = (train_output_root / eval_run_name).resolve()
            checkpoint_override = test.get("checkpoint", "")
            if checkpoint_override:
                best_ckpt = Path(checkpoint_override).expanduser()
                if not best_ckpt.is_absolute():
                    candidate = (_repo_root() / best_ckpt).resolve()
                    best_ckpt = candidate if candidate.exists() else best_ckpt.resolve()
            else:
                best_ckpt = train_run_dir / "best.pt"

            if not best_ckpt.exists():
                raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
            imu_stats_for_eval = _resolve_imu_stats_for_eval(test_cfg, best_ckpt)

        if physics_enabled:
            cmd = [
                sys.executable,
                "-m",
                "src.engine.eval_physics",
                "--config",
                str(config_path),
                "--save_json",
                str(test_run_dir / "physics_results.json"),
            ]
            _run(cmd)
            return state
        else:
            if not dl_enabled:
                raise ValueError("No enabled matcher found under test.matcher. Enable dl_matcher or physics_based_matcher.")

            custom_eval = test.get("mode", "").strip() == "custom_2person"
            if custom_eval:
                test_json = test_run_dir / "eval_results_2person.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "src.engine.eval_custom",
                    "--test_csv", str(paths.get("test_csv", "")),
                    "--data_root", str(paths.get("data_root", "")),
                    "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
                    "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
                    "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
                    "--checkpoint", str(best_ckpt),
                    "--batch_size", str(test.get("batch_size", 64)),
                    "--eval_mode", str(test.get("eval_mode", "same_time_2person")),
                    "--chunk_windows", str(test.get("chunk_windows", 30)),
                    "--save_json", str(test_json),
                ]
                if "imu_sensor" in train_cfg and train_cfg.get("imu_sensor") is not None:
                    cmd.extend(["--imu_sensor", str(train_cfg.get("imu_sensor", ""))])
                if train_cfg.get("repeat_single_sensor") is not None:
                    cmd.extend(["--repeat_single_sensor", str(train_cfg["repeat_single_sensor"])])
                _append_imu_filter_args(cmd, cfg.get("preprocess", {}).get("imu", {}))
                if imu_stats_for_eval is not None:
                    cmd.extend(["--imu_stats_json", str(imu_stats_for_eval)])
                if model.get("use_global_motion"):
                    cmd.append("--use_global_motion")
                if model.get("global_motion_input_dim") is not None:
                    cmd.extend(["--global_motion_input_dim", str(model["global_motion_input_dim"])])
                if model.get("global_motion_hidden_dim") is not None:
                    cmd.extend(["--global_motion_hidden_dim", str(model["global_motion_hidden_dim"])])
                if model.get("global_motion_num_layers") is not None:
                    cmd.extend(["--global_motion_num_layers", str(model["global_motion_num_layers"])])
                if model.get("global_motion_dropout") is not None:
                    cmd.extend(["--global_motion_dropout", str(model["global_motion_dropout"])])
                if model.get("global_motion_input_type"):
                    cmd.extend(["--global_motion_input_type", str(model["global_motion_input_type"])])
                if model.get("global_motion_fusion_type"):
                    cmd.extend(["--global_motion_fusion_type", str(model["global_motion_fusion_type"])])
                if model.get("global_motion_fusion_proj"):
                    cmd.append("--global_motion_fusion_proj")
                if model.get("global_motion_root_source"):
                    cmd.extend(["--global_motion_root_source", str(model["global_motion_root_source"])])
                if model.get("global_motion_aux_weight") is not None:
                    cmd.extend(["--global_motion_aux_weight", str(model["global_motion_aux_weight"])])
                if model.get("global_motion_train_only"):
                    cmd.append("--global_motion_train_only")
                if train_cfg.get("device"):
                    cmd.extend(["--device", str(train_cfg["device"])])
                _run(cmd)
            else:
                test_json = test_run_dir / "test_metrics.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "src.engine.eval",
                    "--test_csv", str(paths.get("test_csv", "")),
                    "--data_root", str(paths.get("data_root", "")),
                    "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
                    "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
                    "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
                    "--checkpoint", str(best_ckpt),
                    "--batch_size", str(test.get("batch_size", 64)),
                    "--save_json", str(test_json),
                ]
                if "imu_sensor" in train_cfg and train_cfg.get("imu_sensor") is not None:
                    cmd.extend(["--imu_sensor", str(train_cfg.get("imu_sensor", ""))])
                if train_cfg.get("repeat_single_sensor") is not None:
                    cmd.extend(["--repeat_single_sensor", str(train_cfg["repeat_single_sensor"])])
                if imu_stats_for_eval is not None:
                    cmd.extend(["--imu_stats_json", str(imu_stats_for_eval)])
                # Physics encoder args for eval
                if model.get("imu_encoder_type"):
                    cmd.extend(["--imu_encoder_type", str(model["imu_encoder_type"])])
                if model.get("physics_d_model") is not None:
                    cmd.extend(["--physics_d_model", str(model["physics_d_model"])])
                if model.get("physics_n_heads") is not None:
                    cmd.extend(["--physics_n_heads", str(model["physics_n_heads"])])
                if model.get("physics_num_layers") is not None:
                    cmd.extend(["--physics_num_layers", str(model["physics_num_layers"])])
                if model.get("physics_fs_hz") is not None:
                    cmd.extend(["--physics_fs_hz", str(model["physics_fs_hz"])])
                if model.get("physics_n_fft") is not None:
                    cmd.extend(["--physics_n_fft", str(model["physics_n_fft"])])
                if model.get("physics_dropout") is not None:
                    cmd.extend(["--physics_dropout", str(model["physics_dropout"])])
                # Global motion encoder args for eval
                if model.get("use_global_motion"):
                    cmd.append("--use_global_motion")
                if model.get("global_motion_input_dim") is not None:
                    cmd.extend(["--global_motion_input_dim", str(model["global_motion_input_dim"])])
                if model.get("global_motion_hidden_dim") is not None:
                    cmd.extend(["--global_motion_hidden_dim", str(model["global_motion_hidden_dim"])])
                if model.get("global_motion_num_layers") is not None:
                    cmd.extend(["--global_motion_num_layers", str(model["global_motion_num_layers"])])
                if model.get("global_motion_dropout") is not None:
                    cmd.extend(["--global_motion_dropout", str(model["global_motion_dropout"])])
                if model.get("global_motion_input_type"):
                    cmd.extend(["--global_motion_input_type", str(model["global_motion_input_type"])])
                if model.get("global_motion_fusion_type"):
                    cmd.extend(["--global_motion_fusion_type", str(model["global_motion_fusion_type"])])
                if model.get("global_motion_fusion_proj"):
                    cmd.append("--global_motion_fusion_proj")
                if model.get("global_motion_root_source"):
                    cmd.extend(["--global_motion_root_source", str(model["global_motion_root_source"])])
                if model.get("global_motion_train_only"):
                    cmd.append("--global_motion_train_only")
                if train_cfg.get("device"):
                    cmd.extend(["--device", str(train_cfg["device"])])
                _run(cmd)

        if grouped.get("enabled", False):
            grouped_json = test_run_dir / "grouped_results.json"
            grouped_csv = test_run_dir / "grouped_results.csv"
            cmd = [
                sys.executable,
                "-m",
                "src.engine.eval_grouped",
                "--test_csv", str(paths.get("test_csv", "")),
                "--data_root", str(paths.get("data_root", "")),
                "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
                "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
                "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
                "--checkpoint", str(best_ckpt),
                "--batch_size", str(test.get("batch_size", 64)),
                "--group_sizes", str(grouped.get("group_sizes", "2,4,6,8,16")),
                "--num_trials", str(grouped.get("num_trials", 50)),
                "--chunk_windows", str(grouped.get("chunk_windows", 30)),
                "--min_chunk_windows", str(grouped.get("min_chunk_windows", 15)),
                "--seed", str(grouped.get("seed", 42)),
                "--save_json", str(grouped_json),
                "--save_csv", str(grouped_csv),
            ]
            if "imu_sensor" in train_cfg and train_cfg.get("imu_sensor") is not None:
                cmd.extend(["--imu_sensor", str(train_cfg.get("imu_sensor", ""))])
            if train_cfg.get("repeat_single_sensor") is not None:
                cmd.extend(["--repeat_single_sensor", str(train_cfg["repeat_single_sensor"])])
            _append_imu_filter_args(cmd, cfg.get("preprocess", {}).get("imu", {}))
            if imu_stats_for_eval is not None:
                cmd.extend(["--imu_stats_json", str(imu_stats_for_eval)])
            # Global motion encoder args for grouped eval
            if model.get("use_global_motion"):
                cmd.append("--use_global_motion")
            if model.get("global_motion_input_dim") is not None:
                cmd.extend(["--global_motion_input_dim", str(model["global_motion_input_dim"])])
            if model.get("global_motion_hidden_dim") is not None:
                cmd.extend(["--global_motion_hidden_dim", str(model["global_motion_hidden_dim"])])
            if model.get("global_motion_num_layers") is not None:
                cmd.extend(["--global_motion_num_layers", str(model["global_motion_num_layers"])])
            if model.get("global_motion_dropout") is not None:
                cmd.extend(["--global_motion_dropout", str(model["global_motion_dropout"])])
            if model.get("global_motion_input_type"):
                cmd.extend(["--global_motion_input_type", str(model["global_motion_input_type"])])
            if model.get("global_motion_fusion_type"):
                cmd.extend(["--global_motion_fusion_type", str(model["global_motion_fusion_type"])])
            if model.get("global_motion_fusion_proj"):
                cmd.append("--global_motion_fusion_proj")
            if model.get("global_motion_root_source"):
                cmd.extend(["--global_motion_root_source", str(model["global_motion_root_source"])])
            if model.get("global_motion_train_only"):
                cmd.append("--global_motion_train_only")
            if train_cfg.get("device"):
                cmd.extend(["--device", str(train_cfg["device"])])
            if grouped.get("per_subject_split"):
                cmd.append("--per_subject_split")
            _run(cmd)

        if synchronous.get("enabled", False):
            sync_json = test_run_dir / "synchronous_results.json"
            cmd = [
                sys.executable,
                "-m",
                "src.engine.eval_synchronous",
                "--test_csv", str(paths.get("test_csv", "")),
                "--data_root", str(paths.get("data_root", "")),
                "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
                "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
                "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
                "--checkpoint", str(best_ckpt),
                "--window_size", str(synchronous.get("window_size", 24)),
                "--stride", str(synchronous.get("stride", 1)),
                "--batch_size", str(test.get("batch_size", 64)),
                "--save_json", str(sync_json),
            ]
            if "imu_sensor" in train_cfg and train_cfg.get("imu_sensor") is not None:
                cmd.extend(["--imu_sensor", str(train_cfg.get("imu_sensor", ""))])
            if train_cfg.get("repeat_single_sensor") is not None:
                cmd.extend(["--repeat_single_sensor", str(train_cfg["repeat_single_sensor"])])
            _append_imu_filter_args(cmd, cfg.get("preprocess", {}).get("imu", {}))
            if imu_stats_for_eval is not None:
                cmd.extend(["--imu_stats_json", str(imu_stats_for_eval)])
            # Global motion encoder args for synchronous eval
            if model.get("use_global_motion"):
                cmd.append("--use_global_motion")
            if model.get("global_motion_input_dim") is not None:
                cmd.extend(["--global_motion_input_dim", str(model["global_motion_input_dim"])])
            if model.get("global_motion_hidden_dim") is not None:
                cmd.extend(["--global_motion_hidden_dim", str(model["global_motion_hidden_dim"])])
            if model.get("global_motion_num_layers") is not None:
                cmd.extend(["--global_motion_num_layers", str(model["global_motion_num_layers"])])
            if model.get("global_motion_dropout") is not None:
                cmd.extend(["--global_motion_dropout", str(model["global_motion_dropout"])])
            if model.get("global_motion_input_type"):
                cmd.extend(["--global_motion_input_type", str(model["global_motion_input_type"])])
            if model.get("global_motion_fusion_type"):
                cmd.extend(["--global_motion_fusion_type", str(model["global_motion_fusion_type"])])
            if model.get("global_motion_fusion_proj"):
                cmd.append("--global_motion_fusion_proj")
            if model.get("global_motion_root_source"):
                cmd.extend(["--global_motion_root_source", str(model["global_motion_root_source"])])
            if model.get("global_motion_train_only"):
                cmd.append("--global_motion_train_only")
            if train_cfg.get("device"):
                cmd.extend(["--device", str(train_cfg["device"])])
            _run(cmd)

        return state


def _run_dir(cfg: dict) -> Path:
    out = cfg.get("train", {}).get("output", {})
    return (_repo_root() / out.get("output_root", "artifacts") / out.get("run_name", "")).resolve()


def _expand_cfg(cfg: dict, fold: int | None) -> dict:
    import copy
    if fold is None:
        return cfg
    return _format_value(copy.deepcopy(cfg), fold)


def _format_value(value: Any, fold: int | None) -> Any:
    if fold is None:
        return value
    if isinstance(value, str):
        try:
            return value.format(fold=fold)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: _format_value(v, fold) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_value(v, fold) for v in value]
    return value
