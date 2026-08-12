from __future__ import annotations

from pathlib import Path

from src.config import load_cfg


def test_load_representative_configs():
    for path in [
        "configs/totalcapture_vicon_test.yaml",
        "configs/egohumans_test.yaml",
        "configs/custom.yaml",
    ]:
        cfg = load_cfg(path)
        assert cfg.PREPROCESS.DATASET
        assert cfg.PATHS.DATA_ROOT


def test_all_official_configs_load():
    paths = sorted(
        path
        for path in Path("configs").rglob("*.yaml")
        if not path.name.startswith("_tmp_")
        and not ({"benchmarks", "detectors", "pose_estimators", "trackers"} & set(path.parts))
    )
    assert paths
    for path in paths:
        load_cfg(path)


def test_unknown_top_level_section_fails_instead_of_being_silently_ignored(tmp_path):
    path = tmp_path / "unknown.yaml"
    path.write_text("project: test\nunknown_domain:\n  enabled: true\n", encoding="utf-8")

    try:
        load_cfg(path)
    except KeyError as exc:
        assert "unknown_domain" in str(exc).lower()
    else:
        raise AssertionError("Unknown top-level config section was accepted")
