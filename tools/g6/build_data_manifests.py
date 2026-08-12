"""Generate deterministic data manifests for every frozen G6 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_manifest import build_prepared_data_manifest
from .matrix import CUSTOM_FOLDS

SOURCE_ROOTS = {
    "totalcapture": Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source"),
    "egohumans": Path("/data/fzliang/reid-project/egohumans/preprocessed/g6_egohumans_source"),
}
CUSTOM_ROOT = Path(
    "/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_rawcsv7d_swapsess"
)
CUSTOM_SEGMENT_ROOT = Path(
    "/data/fzliang/reid-project/custom/evaluation/custom_segments/sequences"
)
CUSTOM_IMU_ROOT = Path("/data/fzliang/data/preprocess/2person")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _custom_evaluation_artifacts(session: str) -> dict[str, Path]:
    artifacts = {
        f"segment/{path.name}": path
        for path in sorted(CUSTOM_SEGMENT_ROOT.glob(f"custom_{session}_seg*.npz"))
    }
    if not artifacts:
        raise FileNotFoundError(f"No Custom segment NPZs found for held-out session {session}")
    session_root = CUSTOM_IMU_ROOT / session
    timestamp_candidates = (
        session_root / "video" / f"{session}_frame_timestamps_retimed.csv",
        session_root / "video" / f"{session}_frame_timestamps.csv",
    )
    timestamp = next((path for path in timestamp_candidates if path.is_file()), None)
    if timestamp is None:
        raise FileNotFoundError(f"No Custom frame timestamp CSV found for {session}")
    artifacts[f"raw_imu/{timestamp.relative_to(CUSTOM_IMU_ROOT)}"] = timestamp
    imu_paths = sorted((session_root / "imu").glob(f"{session}_*.csv"))
    if not imu_paths:
        raise FileNotFoundError(f"No Custom IMU CSVs found for held-out session {session}")
    for path in imu_paths:
        artifacts[f"raw_imu/{path.relative_to(CUSTOM_IMU_ROOT)}"] = path
    return artifacts


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifests = []
    for dataset, root in SOURCE_ROOTS.items():
        manifest = build_prepared_data_manifest(root, dataset=dataset)
        name = f"{dataset}.json"
        _write_json(output_dir / name, manifest)
        manifests.append({"dataset": dataset, "fold_id": None, "file": name, "manifest_hash": manifest["manifest_hash"]})

    for fold_id, fold in CUSTOM_FOLDS.items():
        session = str(fold["test_session"])
        root = CUSTOM_ROOT / f"fold{fold_id}_{session}"
        manifest = build_prepared_data_manifest(
            root,
            dataset="custom",
            fold_id=fold_id,
            evaluation_artifacts=_custom_evaluation_artifacts(session),
        )
        name = f"custom_fold{fold_id}_{session}.json"
        _write_json(output_dir / name, manifest)
        manifests.append({"dataset": "custom", "fold_id": fold_id, "file": name, "manifest_hash": manifest["manifest_hash"]})

    index = {"schema_version": "1.0", "manifests": manifests}
    _write_json(output_dir / "index.json", index)
    print(output_dir / "index.json")


if __name__ == "__main__":
    main()
