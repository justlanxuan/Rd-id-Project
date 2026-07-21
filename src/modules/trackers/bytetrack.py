"""ByteTrack adapter for Autism-project.

Supports both decoupled tracking (BaseTracker) and full subprocess mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.registry import TRACKERS
from src.preprocess.structures import Detection, Track
from src.modules.trackers.base import BaseTracker


@dataclass
class ByteTrackConfig:
    """Configuration for initializing a ByteTrack runtime adapter."""

    repo_root: Optional[str] = "/home/fzliang/Autism-project/third-party/ByteTrack"
    expected_commit: Optional[str] = None
    strict_commit: bool = False
    frame_rate: int = 30
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    mot20: bool = False
    # Subprocess CLI overrides
    exp_file: Optional[str] = "exps/example/mot/yolox_x_mix_det.py"
    ckpt: Optional[str] = "pretrained/bytetrack_x_mot17.pth.tar"
    conf: float = 0.1
    nms: float = 0.7
    tsize: int = 640
    fp16: bool = False
    fuse: bool = False
    device: str = "gpu"
    min_box_area: float = 10.0
    aspect_ratio_thresh: float = 1.6
    python: str = sys.executable


@TRACKERS.register("bytetrack")
class ByteTrackTracker(BaseTracker):
    """Adapter around `yolox.tracker.byte_tracker.BYTETracker`.

    Provides both:
      - `update()`: in-process frame-by-frame tracking (BaseTracker interface)
      - `run_on_video()`: subprocess-based full video tracking
    """

    def __init__(self, config: Optional[ByteTrackConfig] = None):
        self.config = config or ByteTrackConfig()
        self._tracker = None

    def _lazy_init(self) -> None:
        if self._tracker is not None:
            return
        repo_path = self._resolve_repo_path(self.config.repo_root)
        import cv2  # noqa: F401
        if repo_path is not None:
            self._validate_commit(repo_path)
            self._ensure_import_path(repo_path)
        from yolox.tracker.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_thresh=float(self.config.track_thresh),
            track_buffer=int(self.config.track_buffer),
            match_thresh=float(self.config.match_thresh),
            mot20=bool(self.config.mot20),
        )
        self._tracker = BYTETracker(args, frame_rate=int(self.config.frame_rate))

    @staticmethod
    def _resolve_repo_path(repo_root: Optional[str]) -> Optional[Path]:
        if not repo_root:
            return None
        repo_path = Path(repo_root).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"ByteTrack repo not found: {repo_path}")
        return repo_path

    def _validate_commit(self, repo_path: Path) -> None:
        if not self.config.expected_commit:
            return
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if self.config.strict_commit:
                raise RuntimeError(
                    "Failed to read ByteTrack git commit: "
                    f"{result.stderr.strip() or 'unknown error'}"
                )
            return
        actual = result.stdout.strip()
        expected = self.config.expected_commit.strip()
        if actual != expected:
            message = f"ByteTrack commit mismatch: expected {expected}, got {actual}."
            if self.config.strict_commit:
                raise RuntimeError(message)
            print(f"[ByteTrackTracker] Warning: {message}")

    @staticmethod
    def _ensure_import_path(repo_path: Path) -> None:
        repo_root = str(repo_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def reset(self) -> None:
        """Reset tracker state and start a new sequence."""
        self._tracker = None

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Run one tracking update step and return normalized tracking results."""
        self._lazy_init()
        if self._tracker is None:
            raise RuntimeError("ByteTrack runtime is not initialized")

        det = self._detections_to_array(detections)
        height, width = frame.shape[:2]
        online_tracks = self._tracker.update(det, (height, width), (height, width))

        results: List[Track] = []
        for t in online_tracks:
            tlbr = t.tlbr.tolist() if hasattr(t.tlbr, "tolist") else list(t.tlbr)
            detection = Detection(bbox=np.array(tlbr, dtype=np.float32), score=float(getattr(t, "score", 0.0)))
            results.append(Track(track_id=int(t.track_id), detection=detection))
        return results

    @staticmethod
    def _detections_to_array(detections: List[Detection]) -> np.ndarray:
        if not detections:
            return np.zeros((0, 5), dtype=np.float32)
        arr = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox.tolist()
            arr.append([x1, y1, x2, y2, d.score])
        return np.array(arr, dtype=np.float32)

    def run_on_video(self, video_path: str, output_dir: str, env: Optional[Dict[str, str]] = None) -> Path:
        """Run ByteTrack CLI on a video and return the latest track txt path.

        The txt is copied into output_dir for downstream consumption.
        """
        repo_path = self._resolve_repo_path(self.config.repo_root)
        if repo_path is None:
            raise FileNotFoundError("ByteTrack repo_root is required for subprocess mode")

        bt_outdir = Path(output_dir) / "bytetrack_raw"
        bt_outdir.mkdir(parents=True, exist_ok=True)
        bt_track_txt = bt_outdir / "bytetrack_tracks.txt"

        exp_file = self.config.exp_file or "exps/example/mot/yolox_x_mix_det.py"
        ckpt = self.config.ckpt or "pretrained/bytetrack_x_mot17.pth.tar"
        exp_path = Path(exp_file)
        if not exp_path.is_absolute():
            exp_path = repo_path / exp_path
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = repo_path / ckpt_path

        exp_name = exp_path.stem
        bt_track_vis_dir = repo_path / "YOLOX_outputs" / exp_name / "track_vis"

        cmd = [
            self.config.python,
            "tools/demo_track.py",
            "video",
            "--path",
            str(video_path),
            "--save_result",
            "-f",
            str(exp_path),
            "-c",
            str(ckpt_path),
            "--device",
            self.config.device,
            "--conf",
            str(self.config.conf),
            "--nms",
            str(self.config.nms),
            "--tsize",
            str(self.config.tsize),
            "--track_thresh",
            str(self.config.track_thresh),
            "--track_buffer",
            str(self.config.track_buffer),
            "--match_thresh",
            str(self.config.match_thresh),
            "--min_box_area",
            str(self.config.min_box_area),
            "--aspect_ratio_thresh",
            str(self.config.aspect_ratio_thresh),
        ]
        if self.config.fp16:
            cmd.append("--fp16")
        if self.config.fuse:
            cmd.append("--fuse")
        if self.config.mot20:
            cmd.append("--mot20")

        subprocess.run(cmd, check=True, cwd=str(repo_path), env=env)

        txts = sorted(bt_track_vis_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
        if not txts:
            raise FileNotFoundError(f"No tracking txt found under: {bt_track_vis_dir}")
        latest_txt = txts[-1]
        import shutil
        shutil.copy2(latest_txt, bt_track_txt)
        return bt_track_txt
