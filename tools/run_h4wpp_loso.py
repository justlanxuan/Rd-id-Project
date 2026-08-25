#!/usr/bin/env python3
"""Run the four-fold H4W++ Custom three-train/one-test protocol."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ("171423", "171724", "172257", "172522")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=SESSIONS, action="append", help="Held-out session suffix; repeat to select folds.")
    parser.add_argument(
        "--config-prefix",
        default="custom_h4wpp_loso",
        help="Config filename prefix before the session suffix (for example, custom_h4wpp_fullframe_loso).",
    )
    parser.add_argument("--stages", default="preprocess,train,test")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value for the run.")
    args = parser.parse_args()
    selected = tuple(args.fold or SESSIONS)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    for suffix in selected:
        config = REPO_ROOT / "configs" / f"{args.config_prefix}_{suffix}.yaml"
        if not config.is_file():
            raise FileNotFoundError(config)
        command = [
            sys.executable,
            str(REPO_ROOT / "run_pipeline.py"),
            "--config",
            str(config),
            "--stages",
            args.stages,
        ]
        print("[h4wpp-loso]", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
