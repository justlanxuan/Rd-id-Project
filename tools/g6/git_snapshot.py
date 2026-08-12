"""Git snapshot gates for reproducible formal experiments."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_snapshot(repo_root: str | Path) -> tuple[str, list[str]]:
    root = Path(repo_root).expanduser().resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, [line for line in status if line.strip()]


def require_clean_git_snapshot(repo_root: str | Path, expected_commit: str = "") -> str:
    commit, dirty_entries = git_snapshot(repo_root)
    if dirty_entries:
        preview = "; ".join(dirty_entries[:5])
        remainder = len(dirty_entries) - min(len(dirty_entries), 5)
        suffix = f"; plus {remainder} more" if remainder else ""
        raise RuntimeError(
            "Formal G6 execution requires a clean Git worktree so the exact code is reproducible. "
            f"Dirty entries: {preview}{suffix}"
        )
    expected = str(expected_commit).strip()
    if expected and commit != expected:
        raise RuntimeError(
            f"Formal G6 code commit mismatch: current={commit}, protocol={expected}."
        )
    return commit
