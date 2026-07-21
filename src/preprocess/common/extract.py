"""Shared video skeleton extraction runner.

Dataset modules should only discover media and write manifests. This module
owns the common detector/tracker/pose-estimator dispatch for all datasets.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from src.pipelines.video_pipeline.config_loader import assemble_extract_config
from src.pipelines.video_pipeline.video_extractors.base import VideoSkeletonExtractor
from src.utils.config import resolve_config


def iter_manifest_videos(extract_cfg: Dict[str, Any]) -> Iterable[tuple[Path, str | None]]:
    video = extract_cfg.get("video")
    manifest_csv = extract_cfg.get("manifest_csv")
    limit = int(extract_cfg.get("limit", 0))

    if video:
        yield Path(video).expanduser().resolve(), None
        return

    if manifest_csv:
        manifest_path = Path(manifest_csv).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
        with manifest_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                video_path = row.get("video_path", "").strip()
                if not video_path:
                    continue
                count += 1
                if limit and count > limit:
                    break
                result_name = row.get("result_name") or row.get("sequence_id") or None
                yield Path(video_path).expanduser().resolve(), result_name
        return

    raise ValueError("Config extract section must specify 'video' or 'manifest_csv'")


def build_video_skeleton_extractor(extract_cfg: Dict[str, Any]) -> VideoSkeletonExtractor:
    detector = extract_cfg.get("detector")
    tracker = extract_cfg.get("tracker")
    pose_estimator = extract_cfg.get("pose_estimator")

    if detector == "alphapose" and tracker == "alphapose" and pose_estimator == "alphapose":
        from src.pipelines.video_pipeline.video_extractors.alphapose_full import AlphaPoseFullExtractor

        return AlphaPoseFullExtractor(extract_cfg)

    if detector in (None, "bytetrack") and tracker == "bytetrack" and pose_estimator == "alphapose":
        from src.pipelines.video_pipeline.video_extractors.bytetrack_alphapose import ByteTrackAlphaPoseExtractor

        return ByteTrackAlphaPoseExtractor(extract_cfg)

    if pose_estimator == "wham" and detector is None and tracker is None:
        from src.pipelines.video_pipeline.video_extractors.wham import WHAMExtractor

        return WHAMExtractor(extract_cfg)

    raise ValueError(
        f"Unsupported extractor combination: detector={detector}, tracker={tracker}, "
        f"pose_estimator={pose_estimator}. Supported: (alphapose+alphapose+alphapose), "
        f"(bytetrack+bytetrack+alphapose), or (wham)."
    )


def _needs_merge(skeleton_json: Path, cfg: Dict[str, Any]) -> bool:
    merge = cfg.get("merge_tracklets", {})
    if not merge.get("enabled", False):
        return False
    unmerged_json = skeleton_json.with_name("skeleton_unmerged.json")
    return skeleton_json.exists() and not unmerged_json.exists()


def _run_merge(json_path: Path, cfg: Dict[str, Any], dry_run: bool) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "src" / "utils" / "merge_tracklets.py"
    if not script.exists():
        raise FileNotFoundError(f"merge_tracklets script not found: {script}")

    merge = cfg.get("merge_tracklets", {})
    merged_json = json_path.with_name("skeleton_merged.json")
    cmd = [
        sys.executable,
        str(script),
        "--json_path",
        str(json_path),
        "--output_json",
        str(merged_json),
        "--max_gap",
        str(merge.get("max_gap", 10000000)),
        "--score_thresh",
        str(merge.get("score_thresh", 2.2)),
        "--max_norm_dist",
        str(merge.get("max_norm_dist", 2.8)),
        "--max_size_diff",
        str(merge.get("max_size_diff", 1.8)),
    ]
    if merge.get("fill_gaps", False):
        cmd.append("--fill_gaps")
    known = merge.get("known_num_people")
    if known is not None:
        cmd.extend(["--known_num_people", str(known)])

    print(f"\n[CMD] {' '.join(cmd)}")
    if dry_run:
        print("[DRY_RUN] skip merge")
        return
    subprocess.run(cmd, check=True)
    if not merged_json.exists():
        raise FileNotFoundError(f"Merged JSON was not produced: {merged_json}")
    unmerged_json = json_path.with_name("skeleton_unmerged.json")
    shutil.move(str(json_path), str(unmerged_json))
    shutil.move(str(merged_json), str(json_path))
    print(f"[MERGE] Saved unmerged JSON to {unmerged_json}")
    print(f"[MERGE] Overwrote skeleton JSON with merged IDs: {json_path}")


def process_video_skeleton(
    video_path: Path,
    extractor: VideoSkeletonExtractor,
    extract_cfg: Dict[str, Any],
    dry_run: bool = False,
    result_name: str | None = None,
) -> Path:
    results_root = Path(extract_cfg.get(
        "results_root",
        "/data/fzliang/reid-project/extract_outputs",
    )).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    video_name = result_name or video_path.stem
    video_result_dir = results_root / video_name
    video_result_dir.mkdir(parents=True, exist_ok=True)

    skeleton_json = video_result_dir / "skeleton.json"
    skip_existing = extract_cfg.get("skip_existing", False)
    existing_skeleton = skeleton_json.exists()
    merge_needed = _needs_merge(skeleton_json, extract_cfg)

    if skip_existing and existing_skeleton and not merge_needed:
        print(f"[SKIP] {skeleton_json} already exists")
        return skeleton_json

    if existing_skeleton and skip_existing:
        print(f"[SKIP-EXTRACT] Reusing existing {skeleton_json}")
    elif dry_run:
        print(f"[DRY_RUN] Would extract {video_path} -> {video_result_dir}")
    else:
        skeleton_json = Path(extractor.extract(str(video_path), str(video_result_dir)))

    if merge_needed:
        _run_merge(skeleton_json, extract_cfg, dry_run)

    summary = {
        "video": str(video_path),
        "video_name": video_name,
        "video_result_dir": str(video_result_dir),
        "skeleton_json": str(skeleton_json),
        "merge_enabled": extract_cfg.get("merge_tracklets", {}).get("enabled", False),
    }
    summary_path = video_result_dir / "pipeline_run_summary.json"
    if not dry_run:
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
    print(f"\nPipeline finished for {video_name}")
    print(f"Skeleton JSON: {skeleton_json}")
    return skeleton_json


def run_video_skeleton_extraction(config_path: str | Path, dry_run: bool = False) -> None:
    cfg = resolve_config(config_path)
    extract_cfg = cfg.get("extract")
    if not isinstance(extract_cfg, dict):
        print("[INFO] No extract section in config; nothing to do.")
        return

    extract_cfg = assemble_extract_config(extract_cfg)
    videos = list(iter_manifest_videos(extract_cfg))
    if dry_run:
        for video_path, result_name in videos:
            results_root = Path(extract_cfg.get(
                "results_root",
                "/data/fzliang/reid-project/extract_outputs",
            )).expanduser().resolve()
            print(f"[DRY_RUN] Would extract {video_path} -> {results_root / (result_name or video_path.stem)}")
        return

    extractor = build_video_skeleton_extractor(extract_cfg)
    for video_path, result_name in videos:
        process_video_skeleton(video_path, extractor, extract_cfg, dry_run=False, result_name=result_name)
