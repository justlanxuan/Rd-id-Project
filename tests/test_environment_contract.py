from __future__ import annotations

from pathlib import Path

import yaml


def test_environment_declares_core_runtime_and_verification_dependencies() -> None:
    payload = yaml.safe_load(Path("environment.yml").read_text(encoding="utf-8"))

    assert payload["name"] == "reid_project"
    dependencies = payload["dependencies"]
    conda_names = {str(item).split("=", 1)[0] for item in dependencies if isinstance(item, str)}
    pip_section = next(item["pip"] for item in dependencies if isinstance(item, dict) and "pip" in item)
    pip_names = {str(item).split("=", 1)[0] for item in pip_section}

    assert {"python", "pytorch", "pytorch-cuda", "numpy", "scipy", "pyyaml"} <= conda_names
    assert {"yacs", "pytest", "ruff"} <= pip_names
