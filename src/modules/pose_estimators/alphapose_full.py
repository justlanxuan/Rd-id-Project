"""AlphaPose full estimator (detection + tracking + pose) via subprocess.

This is a standalone wrapper that does NOT inherit BasePoseEstimator,
since it operates at the video level via CLI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AlphaPoseFullConfig:
    """Configuration for AlphaPose full subprocess mode."""

    repo_root: str = ""
    cfg_file: Optional[str] = None
    checkpoint_file: Optional[str] = None
    python: str = sys.executable
    detbatch: Optional[int] = None
    posebatch: Optional[int] = None
    gpu: Optional[int] = None
    headless: bool = True
    use_expandable_segments: bool = False


class AlphaPoseFullEstimator:
    """Run AlphaPose demo_inference.py in full video mode (with --pose_track)."""

    def __init__(self, config: Optional[AlphaPoseFullConfig] = None):
        self.config = config or AlphaPoseFullConfig()
        if not self.config.repo_root:
            raise ValueError("AlphaPoseFullConfig.repo_root is required")
        self.repo_path = Path(self.config.repo_root).expanduser().resolve()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"AlphaPose repo not found: {self.repo_path}")

    def run_on_video(self, video_path: str, output_dir: str, env: Optional[Dict[str, str]] = None) -> Path:
        """Run AlphaPose CLI and return path to skeleton.json."""
        ap_outdir = Path(output_dir) / "alphapose_raw"
        ap_outdir.mkdir(parents=True, exist_ok=True)
        json_path = ap_outdir / "alphapose-results.json"
        skeleton_json = Path(output_dir) / "skeleton.json"

        cfg = self.config.cfg_file
        ckpt = self.config.checkpoint_file
        if not cfg or not ckpt:
            raise ValueError("AlphaPoseFullConfig requires cfg_file and checkpoint_file")

        cfg_path = Path(cfg)
        if not cfg_path.is_absolute():
            cfg_path = self.repo_path / cfg_path
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = self.repo_path / ckpt_path

        cmd = [
            self.config.python,
            "scripts/demo_inference.py",
            "--cfg",
            str(cfg_path),
            "--checkpoint",
            str(ckpt_path),
            "--video",
            str(video_path),
            "--outdir",
            str(ap_outdir),
            "--pose_track",
            "--showbox",
            "--save_img",
            "--save_video",
        ]
        if self.config.detbatch is not None:
            cmd.extend(["--detbatch", str(self.config.detbatch)])
        if self.config.posebatch is not None:
            cmd.extend(["--posebatch", str(self.config.posebatch)])

        _env = dict(env) if env is not None else os.environ.copy()
        # Remove the current repository path to avoid conflicting with AlphaPose's internal utils package.
        project_root = Path(__file__).resolve().parents[3]
        filtered_paths = [
            p for p in _env.get("PYTHONPATH", "").split(os.pathsep)
            if p and Path(p).expanduser().resolve() != project_root
        ]
        _env["PYTHONPATH"] = str(self.repo_path)
        if filtered_paths:
            _env["PYTHONPATH"] = f"{self.repo_path}{os.pathsep}{os.pathsep.join(filtered_paths)}"
        if self.config.gpu is not None:
            _env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        else:
            _env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.config.use_expandable_segments:
            _env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if self.config.headless:
            _env.setdefault("MPLBACKEND", "Agg")
            _env.setdefault("QT_QPA_PLATFORM", "offscreen")
            _env.setdefault("SDL_VIDEODRIVER", "dummy")
            _env.setdefault("DISPLAY", "")
            _env.setdefault("HEADLESS", "1")

        subprocess.run(cmd, check=True, cwd=str(self.repo_path), env=_env)

        if not json_path.exists():
            raise FileNotFoundError(f"AlphaPose JSON not found: {json_path}")
        shutil.copy2(json_path, skeleton_json)
        return skeleton_json
