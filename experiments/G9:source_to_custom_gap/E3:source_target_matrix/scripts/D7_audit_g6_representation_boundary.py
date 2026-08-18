# Experiment Note: D7-G6-representation-boundary
"""Prove the current G6 skeleton encoder is invariant to the third coordinate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.modules.encoders.hybrid import raw_pose_sequence, skeleton_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/fzliang/reid-project/g9/e3_source_target/g6_representation_boundary.json"),
    )
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(7)
    xy = torch.rand((2, 24, 17, 2), generator=generator)
    z_a = torch.rand((2, 24, 17, 1), generator=generator)
    z_b = torch.rand((2, 24, 17, 1), generator=generator) * 1000.0 - 500.0
    pose_a = torch.cat((xy, z_a), dim=-1)
    pose_b = torch.cat((xy, z_b), dim=-1)
    raw_a = raw_pose_sequence(pose_a)
    raw_b = raw_pose_sequence(pose_b)
    token_a = skeleton_tokens(pose_a)
    token_b = skeleton_tokens(pose_b)
    result = {
        "schema_version": "g9-e3-g6-representation-boundary-1",
        "input": {
            "shape": list(pose_a.shape),
            "same_xy": bool(torch.equal(pose_a[..., :2], pose_b[..., :2])),
            "different_z": bool(not torch.equal(pose_a[..., 2], pose_b[..., 2])),
        },
        "observed": {
            "raw_pose_max_abs_diff": float(torch.max(torch.abs(raw_a - raw_b))),
            "skeleton_token_max_abs_diff": float(torch.max(torch.abs(token_a - token_b))),
            "raw_pose_equal": bool(torch.equal(raw_a, raw_b)),
            "skeleton_tokens_equal": bool(torch.equal(token_a, token_b)),
        },
        "conclusion": "Current G6 skeleton representation consumes xy only; full-xyz source attribution is not identifiable without a new xyz-compatible encoder/protocol.",
        "source": "src/modules/encoders/hybrid.py::_pose_to_btj2",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **result["observed"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
