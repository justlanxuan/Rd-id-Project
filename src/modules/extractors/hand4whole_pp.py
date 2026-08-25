"""Hand4Whole++ adapter for tracked custom videos."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.modules.extractors.base import ExtractorCapabilities, VideoSkeletonExtractor


class Hand4WholePPExtractor(VideoSkeletonExtractor):
    """Run the official H4W++ model on existing AlphaPose person tracks."""

    capabilities = ExtractorCapabilities(
        output_format="h4wpp_h36m17_json", dimensions=3, experimental=False
    )

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        repo_root = Path(__file__).resolve().parents[3]
        default_root = repo_root / "third-party" / "Hand4Whole-plus-plus_RELEASE"
        default_checkpoint = repo_root / "models" / "hand4whole_plus_plus" / "snapshot_6.pth"
        self.h4w_root = Path(
            cfg.get("h4wpp_root") or os.environ.get("REID_H4WPP_ROOT") or default_root
        ).expanduser().resolve()
        self.checkpoint = Path(
            cfg.get("h4wpp_checkpoint")
            or os.environ.get("REID_H4WPP_CHECKPOINT")
            or default_checkpoint
        ).expanduser().resolve()
        tracks_value = cfg.get("tracks_root") or os.environ.get("REID_TRACKS_ROOT")
        if not tracks_value:
            raise ValueError("Hand4Whole++ requires tracks_root or REID_TRACKS_ROOT")
        self.tracks_root = Path(tracks_value).expanduser().resolve()
        self.python = str(cfg.get("h4wpp_python") or os.environ.get("REID_H4WPP_PYTHON") or sys.executable)
        self.device = str(cfg.get("h4wpp_device", "cuda"))
        self.batch_size = int(cfg.get("h4wpp_batch_size", 2))
        self.script = Path(__file__).resolve().parents[3] / "tools" / "h4wpp_extract_custom.py"

        for label, path in (
            ("h4wpp_root", self.h4w_root),
            ("h4wpp_checkpoint", self.checkpoint),
            ("tracks_root", self.tracks_root),
            ("h4wpp script", self.script),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Hand4Whole++ {label} not found: {path}")

    def extract(self, video_path: str, output_dir: str) -> str:
        video = Path(video_path).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve() / "skeleton.json"
        tracks = self.tracks_root / video.stem / "skeleton.json"
        if not tracks.is_file():
            raise FileNotFoundError(
                f"Hand4Whole++ requires existing AlphaPose tracks for {video.stem}: {tracks}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("HEADLESS", "1")
        env.setdefault("CUDA_VISIBLE_DEVICES", str(self.cfg.get("gpu", 0)))
        python_paths = [
            self.h4w_root / "main",
            self.h4w_root / "common",
            self.h4w_root / "common" / "nets" / "WiLoR",
            self.h4w_root / "common" / "nets" / "mmpose",
        ]
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [*(str(path) for path in python_paths), existing_pythonpath]
            if existing_pythonpath
            else [str(path) for path in python_paths]
        )
        cmd = [
            self.python,
            str(self.script),
            "--h4w-root",
            str(self.h4w_root),
            "--checkpoint",
            str(self.checkpoint),
            "--video",
            str(video),
            "--tracks",
            str(tracks),
            "--output",
            str(output),
            "--device",
            self.device,
            "--batch-size",
            str(self.batch_size),
        ]
        subprocess.run(cmd, check=True, cwd=str(self.h4w_root / "demo"), env=env)
        return str(output)
