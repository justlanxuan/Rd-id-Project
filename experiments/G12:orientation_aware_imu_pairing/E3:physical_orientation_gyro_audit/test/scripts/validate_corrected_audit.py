from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path("/data/fzliang/reid-project/g12/e3_physical_audit/extractor_orientation_imu_join_corrected.json")
METHODS = {"yolopose_high", "alphapose", "fmpose3d", "motionagformer", "tcpformer", "wham"}


def main() -> None:
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert data["schema_version"] == "g12.e3.extractor_orientation_imu_join.v2"
    assert data["failures"] == []
    assert data["no_pairing_ablation"] is True
    assert set(data["summary"]) == METHODS
    assert len(data["records"]) == 6 * 313
    for method, row in data["summary"].items():
        assert row["tracks"] == 313, method
        assert row["sequences"] == 88, method
        control = row["person_shuffle_control"]
        assert control["matched_median_abs_r"] > control["shuffled_median_abs_r"], method
        assert control["matched_gt_shuffled_fraction"] > 0.75, method
        assert row["median_abs_best_axis_r_smooth_0p25s"] > row["time_shuffle_control_median_abs_r95"], method
    print("PASS corrected extractor-orientation × external-IMU audit")


if __name__ == "__main__":
    main()
