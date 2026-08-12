"""AlphaPose tracker adapter for Re-id-Project.

Wraps AlphaPose's original tracker implementation without modifying the
AlphaPose source tree.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np

from src.data import Detection, Track
from src.modules.trackers.base import BaseTracker


@dataclass
class AlphaPoseTrackerConfig:
    repo_root: str = ""
    expected_commit: Optional[str] = None
    strict_commit: bool = False
    tracker_cfg: Optional[Any] = None
    device: str = "cuda:0"
    gpus: Optional[list[int]] = None


class AlphaPoseTracker(BaseTracker):
    """In-process AlphaPose tracker (for future decoupled usage)."""

    def __init__(self, config: Optional[AlphaPoseTrackerConfig] = None):
        self.config = config or AlphaPoseTrackerConfig()
        self.repo_path = self._resolve_repo_path(self.config.repo_root)
        self._tracker = None

    def _lazy_init(self) -> None:
        if self._tracker is not None:
            return
        self._validate_commit(self.repo_path)
        self._ensure_import_path(self.repo_path)
        from trackers.tracker_api import Tracker
        from trackers.tracker_cfg import cfg as tcfg

        tracker_cfg = self.config.tracker_cfg or tcfg
        runtime_args = SimpleNamespace(
            device=self.config.device,
            gpus=self.config.gpus or ([0] if self.config.device.startswith("cuda") else [-1]),
        )
        self._tracker = Tracker(tracker_cfg, runtime_args)

    @staticmethod
    def _resolve_repo_path(repo_root: str) -> Path:
        if not repo_root:
            raise ValueError("AlphaPoseTrackerConfig.repo_root is required")
        repo_path = Path(repo_root).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"AlphaPose repo not found: {repo_path}")
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
                raise RuntimeError(result.stderr.strip() or "Failed to read AlphaPose commit")
            return
        actual = result.stdout.strip()
        expected = self.config.expected_commit.strip()
        if actual != expected:
            message = f"AlphaPose commit mismatch: expected {expected}, got {actual}."
            if self.config.strict_commit:
                raise RuntimeError(message)
            print(f"[AlphaPoseTracker] Warning: {message}")

    @staticmethod
    def _ensure_import_path(repo_path: Path) -> None:
        repo_root = str(repo_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def reset(self) -> None:
        """AlphaPose's original tracker is sequence-local; rebuild it by re-init."""
        self._tracker = None

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker state with detections.

        Note: AlphaPose tracker expects a specific internal format;
        this method is a shim for future decoupled usage.
        """
        self._lazy_init()
        # AlphaPose tracker update signature differs from our generic interface.
        # For now, delegate directly; users needing full control should use
        # AlphaPoseFullEstimator subprocess path.
        raw = self._tracker.update(detections, frame)
        # Normalize to Track structures if possible
        return raw
