from __future__ import annotations

import pytest

from src.modules.extractors import assemble_extract_config


def test_component_fragments_are_parsed_separately_from_workflow_yaml(monkeypatch):
    monkeypatch.setenv("BYTETRACK_ROOT", "/opt/bytetrack")
    monkeypatch.setenv("BYTETRACK_CKPT", "/opt/weights/bytetrack.pth")
    monkeypatch.setenv("ALPHAPOSE_ROOT", "/opt/alphapose")
    monkeypatch.setenv("ALPHAPOSE_CKPT", "/opt/weights/alphapose.pth")
    config = assemble_extract_config(
        {
            "detector": "yolox",
            "tracker": "bytetrack",
            "pose_estimator": "alphapose",
            "gpu": 3,
        }
    )

    assert config["gpu"] == 3
    assert config["bytetrack_root"] == "/opt/bytetrack"
    assert config["alphapose_root"] == "/opt/alphapose"


def test_component_fragment_missing_machine_variable_fails_loud(monkeypatch):
    monkeypatch.delenv("BYTETRACK_ROOT", raising=False)

    with pytest.raises(ValueError, match="BYTETRACK_ROOT"):
        assemble_extract_config({"detector": "yolox"})
