from __future__ import annotations

import subprocess
import sys


def test_legacy_benchmark_runner_fails_with_g6_migration_commands():
    result = subprocess.run(
        [sys.executable, "tools/run_benchmark.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "archived pre-G6 protocol" in result.stderr
    assert "python -m tools.g6.run_jobs --help" in result.stderr
