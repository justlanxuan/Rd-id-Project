"""ByteTrack subprocess adapter used by the composed video extractor."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ByteTrackConfig:
    """Configuration for initializing a ByteTrack runtime adapter."""

    repo_root: Optional[str] = None
    expected_commit: Optional[str] = None
    strict_commit: bool = False
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


class ByteTrackTracker:
    """Run ByteTrack's supported full-video command-line workflow."""

    def __init__(self, config: Optional[ByteTrackConfig] = None):
        self.config = config or ByteTrackConfig()

    @staticmethod
    def _resolve_repo_path(repo_root: Optional[str]) -> Optional[Path]:
        if not repo_root:
            raise ValueError("ByteTrackConfig.repo_root is required")
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

    def run_on_video(self, video_path: str, output_dir: str, env: Optional[Dict[str, str]] = None) -> Path:
        """Run ByteTrack CLI on a video and return the latest track txt path.

        The txt is copied into output_dir for downstream consumption.
        """
        repo_path = self._resolve_repo_path(self.config.repo_root)
        if repo_path is None:
            raise FileNotFoundError("ByteTrack repo_root is required for subprocess mode")
        self._validate_commit(repo_path)

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
