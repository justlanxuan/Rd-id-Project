"""Subprocess helpers shared by workflow stages."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    root = str(repo_root())
    src = str(repo_root() / "src")
    current = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (root, src, current) if part)
    return env


def run_command(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(
        command,
        check=True,
        cwd=str(repo_root()),
        env=env_with_pythonpath(),
    )
