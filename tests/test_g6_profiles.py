from __future__ import annotations

import json

from src.config import load_cfg
from tools.g6.configs import generate_resolved_configs
from tools.g6.profiles import get_profile


def _data_index(path):
    rows = [
        {"dataset": "totalcapture", "fold_id": None, "manifest_hash": "tc"},
        {"dataset": "egohumans", "fold_id": None, "manifest_hash": "ego"},
    ]
    rows.extend(
        {"dataset": "custom", "fold_id": fold, "manifest_hash": f"custom-{fold}"}
        for fold in range(1, 5)
    )
    path.write_text(json.dumps({"manifests": rows}), encoding="utf-8")


def test_stride24_profile_generates_only_24_stride_configs(tmp_path):
    index = tmp_path / "index.json"
    _data_index(index)
    rows = generate_resolved_configs(
        tmp_path / "configs",
        protocol_hash="a" * 64,
        data_manifest_index=index,
        artifact_root=tmp_path / "artifacts",
        profile_name="stride24",
    )

    assert len(rows) == 108
    for row in rows:
        cfg = load_cfg(tmp_path / "configs" / row["config"])
        assert cfg.SLICE.WINDOW_LEN == 24
        assert cfg.SLICE.STRIDE == 24
        if cfg.PREPROCESS.DATASET == "custom":
            assert cfg.TEST.METRICS.FRAME_ACC.WINDOW_SIZE == 24
            assert cfg.TEST.METRICS.FRAME_ACC.STRIDE == 24
            assert str(get_profile("stride24").custom_root) in cfg.PREPROCESS.PREPARED_ROOT
