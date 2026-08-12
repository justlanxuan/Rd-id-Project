from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocess.common.extract import process_video_skeleton, resolve_extract_config


def _valid_entries():
    return [{"image_id": "00000.jpg", "keypoints": [1.0] * 51, "idx": 0}]


class RecordingExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, video_path: str, output_dir: str) -> str:
        del video_path
        self.calls += 1
        output = Path(output_dir) / "skeleton.json"
        output.write_text(json.dumps(_valid_entries()), encoding="utf-8")
        return str(output)


def test_valid_existing_skeleton_is_validated_and_reused(tmp_path):
    result_dir = tmp_path / "results" / "sample"
    result_dir.mkdir(parents=True)
    skeleton = result_dir / "skeleton.json"
    skeleton.write_text(json.dumps(_valid_entries()), encoding="utf-8")
    extractor = RecordingExtractor()

    result = process_video_skeleton(
        tmp_path / "sample.mp4",
        extractor,
        {"results_root": str(tmp_path / "results"), "reuse_existing": True},
        result_name="sample",
    )

    assert result == skeleton
    assert extractor.calls == 0


def test_invalid_existing_skeleton_fails_instead_of_silent_reuse(tmp_path):
    result_dir = tmp_path / "results" / "sample"
    result_dir.mkdir(parents=True)
    (result_dir / "skeleton.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        process_video_skeleton(
            tmp_path / "sample.mp4",
            RecordingExtractor(),
            {"results_root": str(tmp_path / "results"), "reuse_existing": True},
            result_name="sample",
        )


def test_invalid_cache_can_be_explicitly_reextracted(tmp_path):
    result_dir = tmp_path / "results" / "sample"
    result_dir.mkdir(parents=True)
    skeleton = result_dir / "skeleton.json"
    skeleton.write_text("[]", encoding="utf-8")
    extractor = RecordingExtractor()

    result = process_video_skeleton(
        tmp_path / "sample.mp4",
        extractor,
        {
            "results_root": str(tmp_path / "results"),
            "reuse_existing": True,
            "invalid_cache_policy": "reextract",
        },
        result_name="sample",
    )

    assert result == skeleton
    assert extractor.calls == 1


def test_public_top_level_extract_config_is_not_ignored_by_preprocess():
    config = {
        "extract": {"detector": "alphapose", "force": True, "gpu": 1},
        "preprocess": {"extract": {"gpu": 2}},
    }

    resolved = resolve_extract_config(config)

    assert resolved == {"detector": "alphapose", "force": True, "gpu": 2}
    assert resolve_extract_config({"preprocess": {}}) is None
