"""AlphaPose full video extractor (detection + tracking + pose via subprocess)."""

from __future__ import annotations

import sys
from typing import Any, Dict

from src.modules.extractors.base import ExtractorCapabilities, VideoSkeletonExtractor
from src.modules.pose_estimators.alphapose_full import AlphaPoseFullConfig, AlphaPoseFullEstimator


class AlphaPoseFullExtractor(VideoSkeletonExtractor):
    capabilities = ExtractorCapabilities(output_format="alphapose_json", dimensions=2)

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.estimator = AlphaPoseFullEstimator(
            AlphaPoseFullConfig(
                repo_root=cfg.get("pose_estimator_root", cfg.get("alphapose_root", "")),
                cfg_file=cfg.get("pose_estimator_cfg", cfg.get("cfg_file", cfg.get("alphapose_cfg"))),
                checkpoint_file=cfg.get("pose_estimator_ckpt", cfg.get("checkpoint_file", cfg.get("alphapose_ckpt"))),
                python=cfg.get("alphapose_python", sys.executable),
                detbatch=cfg.get("detbatch"),
                posebatch=cfg.get("posebatch"),
                gpu=cfg.get("gpu"),
                headless=cfg.get("headless", True),
                use_expandable_segments=cfg.get("use_expandable_segments", False),
            )
        )

    def extract(self, video_path: str, output_dir: str) -> str:
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.estimator.repo_path)
        if self.estimator.config.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.estimator.config.gpu)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.estimator.config.headless:
            env.setdefault("MPLBACKEND", "Agg")
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            env.setdefault("SDL_VIDEODRIVER", "dummy")
            env.setdefault("DISPLAY", "")
            env.setdefault("HEADLESS", "1")
        if self.estimator.config.use_expandable_segments:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        skeleton_json = self.estimator.run_on_video(video_path, output_dir, env=env)
        return str(skeleton_json)
