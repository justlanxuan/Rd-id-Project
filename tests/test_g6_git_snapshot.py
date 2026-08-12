from __future__ import annotations

import subprocess

import pytest

from tools.g6.git_snapshot import require_clean_git_snapshot


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_formal_git_snapshot_requires_clean_matching_commit(tmp_path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "g6-test@example.invalid")
    _git(tmp_path, "config", "user.name", "G6 Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "snapshot")
    commit = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()

    assert require_clean_git_snapshot(tmp_path, expected_commit=commit) == commit

    with pytest.raises(RuntimeError, match="commit mismatch"):
        require_clean_git_snapshot(tmp_path, expected_commit="f" * 40)

    tracked.write_text("v2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        require_clean_git_snapshot(tmp_path, expected_commit=commit)
