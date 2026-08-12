"""WHAM 3D video skeleton extractor."""

from __future__ import annotations

from typing import Any, Dict

from src.modules.estimators import build_wham_3d_estimator
from src.modules.extractors.base import ExtractorCapabilities, VideoSkeletonExtractor


class WHAMExtractor(VideoSkeletonExtractor):
    capabilities = ExtractorCapabilities(output_format="wham_pickle", dimensions=3, experimental=True)

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        wham_cfg = {
            "repo_root": cfg.get("pose_estimator_root", ""),
            "checkpoint_file": cfg.get("pose_estimator_ckpt"),
            "device": "cuda:0" if cfg.get("gpu") is not None else "cpu",
            "run_global": cfg.get("wham_run_global", True),
            "output_dir": cfg.get("results_root", "./wham_outputs"),
        }
        self.estimator = build_wham_3d_estimator(wham_cfg)

    def extract(self, video_path: str, output_dir: str) -> str:
        self.estimator.process_video(video_path, output_dir=output_dir)
        return f"{output_dir}/wham_results.pkl"
