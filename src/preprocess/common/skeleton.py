"""Shared skeleton extraction helpers.

This module centralizes two things:
- reading AlphaPose-style skeleton JSON files
- thin extraction entrypoints backed by src.modules pose estimators
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .alphapose import (
    find_skeleton_for_sequence,
    load_alphapose_multiperson,
    load_alphapose_skeleton,
)


def run_alphapose_full(video_path: str, output_dir: str, cfg: dict[str, Any] | None = None, env: dict[str, str] | None = None) -> Path:
    from src.modules.pose_estimators.alphapose_full import AlphaPoseFullConfig, AlphaPoseFullEstimator

    estimator = AlphaPoseFullEstimator(AlphaPoseFullConfig(**(cfg or {})))
    return estimator.run_on_video(video_path, output_dir, env=env)


def run_alphapose_sppe(
    video_path: str,
    output_dir: str,
    detfile: str,
    cfg: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    from src.modules.pose_estimators.alphapose_sppe import AlphaPoseSPPE, AlphaPoseSPPEConfig

    estimator = AlphaPoseSPPE(AlphaPoseSPPEConfig(**(cfg or {})))
    return estimator.run_on_video(video_path, output_dir, detfile=detfile, env=env)


def run_wham_3d(video_path: str, output_dir: str | None = None, cfg: dict[str, Any] | None = None) -> dict:
    from src.modules.pose_estimators.wham_3d import build_wham_3d_estimator

    estimator = build_wham_3d_estimator(cfg or {})
    return estimator.process_video(video_path, output_dir=output_dir)


def extract_skeleton(
    video_path: str,
    output_dir: str,
    method: str = "alphapose_full",
    cfg: dict[str, Any] | None = None,
    detfile: str | None = None,
    env: dict[str, str] | None = None,
) -> Path | dict:
    method = method.strip().lower()
    if method in {"alphapose", "alphapose_full", "full"}:
        return run_alphapose_full(video_path, output_dir, cfg=cfg, env=env)
    if method in {"alphapose_sppe", "sppe", "detfile"}:
        if detfile is None:
            raise ValueError("detfile is required for AlphaPose SPPE extraction")
        return run_alphapose_sppe(video_path, output_dir, detfile=detfile, cfg=cfg, env=env)
    if method in {"wham", "wham_3d"}:
        return run_wham_3d(video_path, output_dir, cfg=cfg)
    raise ValueError(f"Unknown skeleton extraction method: {method}")