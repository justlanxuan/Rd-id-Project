# Experiment Note: E5-A2-O2D-alpha-pose-proxy-control
"""Run the 2D extractor-derived orientation control for E5."""

from __future__ import annotations

from E5_run_screen import _args

from tools.g12.train_orientation_matcher import main as train_main


def main() -> int:
    for regime in ("eh_only", "tc_only", "tc_eh_balanced"):
        for variant, profile in (("baseline", "full"), ("turning_cross", "full"), ("turning_cross", "rate")):
            for seed in range(3):
                train_main(
                    _args(
                        regime,
                        variant,
                        profile,
                        seed,
                        orientation_mode="proxy",
                        output_subdir="screen_fully_aligned_o2d",
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
