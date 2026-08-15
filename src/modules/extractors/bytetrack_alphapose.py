"""Composed extractor: ByteTrack (subprocess) + AlphaPose SPPE (subprocess detfile mode)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

from src.modules.extractors.base import ExtractorCapabilities, VideoSkeletonExtractor
from src.modules.extractors.utils import convert_bytetrack_txt_to_detfile, extract_video_frames
from src.modules.pose_estimators.alphapose_sppe import AlphaPoseSPPE, AlphaPoseSPPEConfig
from src.modules.trackers.bytetrack import ByteTrackConfig, ByteTrackTracker


class ByteTrackAlphaPoseExtractor(VideoSkeletonExtractor):
    capabilities = ExtractorCapabilities(output_format="alphapose_json", dimensions=2)

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.tracker = ByteTrackTracker(
            ByteTrackConfig(
                repo_root=cfg.get("tracker_root", cfg.get("bytetrack_root", "")),
                ckpt=cfg.get("tracker_ckpt", cfg.get("bytetrack_ckpt")),
                exp_file=cfg.get("tracker_cfg", cfg.get("bytetrack_exp_file")),
                python=cfg.get("bytetrack_python", sys.executable),
                device="gpu" if cfg.get("gpu") is not None else "cpu",
                conf=cfg.get("bytetrack_conf", 0.1),
                nms=cfg.get("bytetrack_nms", 0.7),
                tsize=cfg.get("bytetrack_tsize", 640),
                fp16=cfg.get("bytetrack_fp16", False),
                fuse=cfg.get("bytetrack_fuse", False),
                mot20=cfg.get("bytetrack_mot20", False),
                track_thresh=cfg.get("bt_track_thresh", 0.5),
                track_buffer=cfg.get("bt_track_buffer", 30),
                match_thresh=cfg.get("bt_match_thresh", 0.8),
                min_box_area=cfg.get("bytetrack_min_box_area", 10.0),
                aspect_ratio_thresh=cfg.get("bytetrack_aspect_ratio_thresh", 1.6),
            )
        )
        self.pose_estimator = AlphaPoseSPPE(
            AlphaPoseSPPEConfig(
                repo_root=cfg.get("pose_estimator_root", cfg.get("alphapose_root", "")),
                cfg_file=cfg.get("pose_estimator_cfg", cfg.get("cfg_file", cfg.get("alphapose_cfg"))),
                checkpoint_file=cfg.get("pose_estimator_ckpt", cfg.get("checkpoint_file", cfg.get("alphapose_ckpt"))),
                python=cfg.get("alphapose_python", "python"),
                detbatch=cfg.get("detbatch"),
                posebatch=cfg.get("posebatch"),
                gpu=cfg.get("gpu"),
                headless=cfg.get("headless", True),
                use_expandable_segments=cfg.get("use_expandable_segments", False),
            )
        )

    def extract(self, video_path: str, output_dir: str) -> str:

        video_path_obj = Path(video_path).expanduser().resolve()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if self.cfg.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cfg["gpu"])
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.cfg.get("headless", True):
            env.setdefault("MPLBACKEND", "Agg")
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            env.setdefault("SDL_VIDEODRIVER", "dummy")
            env.setdefault("DISPLAY", "")
            env.setdefault("HEADLESS", "1")

        bt_repo = str(self.tracker.config.repo_root)
        existing_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = bt_repo
        if existing_py:
            env["PYTHONPATH"] = f"{bt_repo}{os.pathsep}{existing_py}"

        track_txt = self.tracker.run_on_video(str(video_path_obj), str(out_dir), env=env)

        ap_repo = str(self.pose_estimator.repo_path)
        env["PYTHONPATH"] = ap_repo
        if existing_py:
            env["PYTHONPATH"] = f"{ap_repo}{os.pathsep}{existing_py}"

        bt_outdir = out_dir / "bytetrack_raw"
        bt_frame_dir = bt_outdir / "frames"
        bt_detfile_json = bt_outdir / "bytetrack_detfile.json"
        extract_video_frames(video_path_obj, bt_frame_dir)
        convert_bytetrack_txt_to_detfile(track_txt, bt_frame_dir, bt_detfile_json)

        skeleton_json = self.pose_estimator.run_on_video(
            str(video_path_obj), str(out_dir), detfile=str(bt_detfile_json), env=env
        )
        return str(skeleton_json)
