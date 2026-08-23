#!/usr/bin/env python3
"""Validate the extractor-focused G12 E1 inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "yolopose_high": "2d_joints_derived_proxy",
    "alphapose": "2d_joints_derived_proxy",
    "fmpose3d": "3d_joints_derived_heading",
    "motionagformer": "3d_joints_derived_heading",
    "tcpformer": "3d_joints_derived_heading",
    "wham": "direct_orientation_raw_but_not_canonical",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    assert payload["schema_version"] == "g12.extractor_orientation_inventory.v1"
    assert payload["scope"].startswith("our extractor artifacts")
    records = {record["method"]: record for record in payload["records"]}
    assert set(records) == set(EXPECTED), sorted(records)
    for method, expected_classification in EXPECTED.items():
        record = records[method]
        assert record["classification"] == expected_classification, (method, record["classification"])
        assert record["canonical_npz_count"] == 304, (method, record["canonical_npz_count"])
        assert record["algorithm_npz_count"] == 88, (method, record["algorithm_npz_count"])
        assert record["full_scan"]["canonical"]["errors"] == 0, method
        assert record["full_scan"]["canonical"]["nonfinite_skeleton_files"] == 0, method
        assert record["full_scan"]["canonical"]["orientation_key_files"] == 0, method
        assert record["full_scan"]["algorithm"]["errors"] == 0, method
        assert record["full_scan"]["algorithm"]["nonfinite_skeleton_files"] == 0, method
        assert record["full_scan"]["algorithm"]["orientation_key_files"] == 0, method
        assert not record["orientation_like_keys"]["canonical"], method
        sample = record["sample_algorithm"]
        assert sample and "error" not in sample, method
        assert sample["arrays"]["skeleton"]["shape"][-2:] == [17, 3], method
        assert sample["arrays"]["skeleton"]["finite_fraction"] == 1.0, method
    wham = records["wham"]
    assert "root_orient" in wham["orientation_like_keys"]["raw"]
    for method in ("fmpose3d", "motionagformer", "tcpformer"):
        metadata = records[method]["sample_algorithm"].get("metadata", {})
        assert "root-centered torso-scaled H36M17 xyz" in metadata.get("output", ""), method
    print(f"PASS extractor inventory: {len(records)} methods")


if __name__ == "__main__":
    main()
