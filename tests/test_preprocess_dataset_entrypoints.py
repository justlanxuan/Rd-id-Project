from pathlib import Path

from preprocess.adapters import build_dataset_adapter
from preprocess.common.config import load_section_config, parse_bool
from preprocess.common.extract import coco_to_h36m17, run_extraction_if_enabled
from preprocess.datasets import custom, egohumans, totalcapture
from src.modules.estimators import AlphaPoseFullEstimator, AlphaPoseSPPE, WHAM3DEstimator


def test_dataset_entrypoints_expose_runtime_helpers():
    assert callable(custom.run_preprocess)
    assert callable(custom.main)
    assert callable(egohumans.run_preprocess)
    assert callable(egohumans.main)
    assert callable(totalcapture.run_preprocess)
    assert callable(totalcapture.main)


def test_official_adapter_requires_an_existing_config(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError, match="config not found"):
        build_dataset_adapter("custom", tmp_path / "missing.yaml")


def test_extraction_helper_can_be_disabled():
    assert run_extraction_if_enabled(None, None, {"enabled": False}) is None


def test_common_helpers_expose_shared_parsing_defaults(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("preprocess:\n  camera: cam2\n  skeleton_normalize: 'yes'\n", encoding="utf-8")
    cfg = load_section_config(str(config_path), "preprocess")
    assert cfg.get("camera") == "cam2"
    assert parse_bool(cfg.get("skeleton_normalize"), default=False) is True


def test_common_alpha_pose_conversion_shape():
    import numpy as np

    coco = np.zeros((2, 17, 3), dtype=np.float32)
    h36m = coco_to_h36m17(coco)
    assert h36m.shape == (2, 17, 3)


def test_unified_estimators_package_exports_core_backends():
    assert AlphaPoseFullEstimator is not None
    assert AlphaPoseSPPE is not None
    assert WHAM3DEstimator is not None
