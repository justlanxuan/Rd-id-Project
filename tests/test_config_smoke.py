from __future__ import annotations

from src.config import load_cfg


def test_load_representative_configs():
    for path in [
        "configs/totalcapture_vicon_test.yaml",
        "configs/egohumans_test.yaml",
        "configs/custom.yaml",
    ]:
        cfg = load_cfg(path)
        assert cfg.PREPROCESS.DATASET
        assert cfg.PATHS.DATA_ROOT
