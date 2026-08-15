"""AlphaPose pose-only subprocess adapter for tracked detections."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class AlphaPoseSPPEConfig:
    """Configuration for AlphaPose's detfile command-line mode."""

    repo_root: str = ""
    cfg_file: Optional[str] = None
    checkpoint_file: Optional[str] = None
    python: str = sys.executable
    detbatch: Optional[int] = None
    posebatch: Optional[int] = None
    gpu: Optional[int] = None
    headless: bool = True
    use_expandable_segments: bool = False


class AlphaPoseSPPE:
    """Run AlphaPose on bounding boxes produced by the ByteTrack extractor."""

    def __init__(self, config: Optional[AlphaPoseSPPEConfig] = None):
        self.config = config or AlphaPoseSPPEConfig()
        if not self.config.repo_root:
            raise ValueError("AlphaPoseSPPEConfig.repo_root is required")
        self.repo_path = Path(self.config.repo_root).expanduser().resolve()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"AlphaPose repo not found: {self.repo_path}")

    def run_on_video(
        self,
        video_path: str,
        output_dir: str,
        detfile: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Run AlphaPose CLI in detfile mode and return ``skeleton.json``."""
        del video_path
        if detfile is None:
            raise ValueError("AlphaPoseSPPE subprocess mode requires a detfile")

        ap_outdir = Path(output_dir) / "alphapose_raw"
        ap_outdir.mkdir(parents=True, exist_ok=True)
        json_path = ap_outdir / "alphapose-results.json"
        skeleton_json = Path(output_dir) / "skeleton.json"

        cfg = self.config.cfg_file
        ckpt = self.config.checkpoint_file
        if not cfg or not ckpt:
            raise ValueError("AlphaPoseSPPEConfig requires cfg_file and checkpoint_file")

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
            "--detfile",
            str(detfile),
            "--outdir",
            str(ap_outdir),
        ]
        if self.config.detbatch is not None:
            cmd.extend(["--detbatch", str(self.config.detbatch)])
        if self.config.posebatch is not None:
            cmd.extend(["--posebatch", str(self.config.posebatch)])

        runtime_env = dict(env) if env is not None else os.environ.copy()
        project_root = Path(__file__).resolve().parents[3]
        external_paths = [
            path
            for path in runtime_env.get("PYTHONPATH", "").split(os.pathsep)
            if path and Path(path).expanduser().resolve() != project_root
        ]
        alphapose_paths = [
            str(self.repo_path),
            str(self.repo_path / "trackers"),
            str(self.repo_path / "detector" / "tracker"),
        ]
        runtime_env["PYTHONPATH"] = os.pathsep.join([*alphapose_paths, *external_paths])
        if self.config.gpu is not None:
            runtime_env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        else:
            runtime_env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.config.use_expandable_segments:
            runtime_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if self.config.headless:
            runtime_env.setdefault("MPLBACKEND", "Agg")
            runtime_env.setdefault("QT_QPA_PLATFORM", "offscreen")
            runtime_env.setdefault("SDL_VIDEODRIVER", "dummy")
            runtime_env.setdefault("DISPLAY", "")
            runtime_env.setdefault("HEADLESS", "1")

        subprocess.run(cmd, check=True, cwd=str(self.repo_path), env=runtime_env)

        if not json_path.exists():
            raise FileNotFoundError(f"AlphaPose JSON not found: {json_path}")
        shutil.copy2(json_path, skeleton_json)
        return skeleton_json
