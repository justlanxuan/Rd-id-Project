from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import run_pipeline as pipeline


def test_public_pipeline_script_is_directly_executable() -> None:
    script = Path("run_pipeline.py").resolve()

    assert os.access(script, os.X_OK)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), environment.get("PATH", "")])
    completed = subprocess.run(
        [str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "--config" in completed.stdout
    assert "--stages" in completed.stdout


def test_public_stage_parser_defaults_and_composition():
    assert pipeline.parse_stages("all") == ["preprocess", "train", "test"]
    assert pipeline.parse_stages("preprocess") == ["preprocess"]
    assert pipeline.parse_stages("train,test") == ["train", "test"]
    assert pipeline.parse_stages("preprocess,train,test") == ["preprocess", "train", "test"]


def test_legacy_stage_aliases_are_explicitly_deprecated():
    with pytest.warns(DeprecationWarning, match="prepare"):
        assert pipeline.parse_stages("prepare") == ["preprocess"]
    with pytest.warns(DeprecationWarning, match="evaluate"):
        assert pipeline.parse_stages("evaluate") == ["test"]


@pytest.mark.parametrize("spec", ["", ",", "extract", "preprocess,prepare"])
def test_invalid_or_duplicate_public_stages_fail(spec):
    with pytest.raises(ValueError):
        pipeline.parse_stages(spec)


def test_default_pipeline_runs_three_public_stages_in_order(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: contract-test\n", encoding="utf-8")
    calls: list[str] = []

    def stage(name: str):
        def run(config, state):
            calls.append(name)
            state = dict(state)
            state[name] = str(config)
            return state

        return run

    monkeypatch.setattr(
        pipeline,
        "STAGE_FUNCS",
        {name: stage(name) for name in pipeline.DEFAULT_STAGES},
    )
    state = pipeline.run_pipeline(config_path)

    assert calls == ["preprocess", "train", "test"]
    assert list(key for key in state if key != "config_path") == calls


def test_pipeline_can_run_one_stage_and_rejects_missing_config(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: contract-test\n", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "STAGE_FUNCS",
        {
            "preprocess": lambda config, state: state,
            "train": lambda config, state: calls.append("train") or state,
            "test": lambda config, state: state,
        },
    )

    pipeline.run_pipeline(config_path, ["train"])
    assert calls == ["train"]
    with pytest.raises(FileNotFoundError, match="config not found"):
        pipeline.run_pipeline(tmp_path / "missing.yaml")
