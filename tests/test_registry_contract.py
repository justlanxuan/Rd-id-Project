from __future__ import annotations

import pytest

from src.core import Registry
from src.modules.extractors import EXTRACTOR_REGISTRY, build_extractor
from src.modules.pose_estimators.alphapose_full import AlphaPoseFullConfig, AlphaPoseFullEstimator
from src.modules.trackers.bytetrack import ByteTrackConfig, ByteTrackTracker


def test_registry_is_domain_scoped_and_fail_loud():
    registry: Registry[dict] = Registry("example")

    @registry.register("first", aliases=("one",))
    def build_first(value=1):
        return {"value": value}

    assert registry.names() == ("first",)
    assert registry.resolve_name("ONE") == "first"
    assert registry.build("first", value=2) == {"value": 2}
    with pytest.raises(KeyError, match="Available: first"):
        registry.build("missing")
    with pytest.raises(KeyError, match="Duplicate"):
        registry.register("first")(lambda: {})


def test_extractor_registry_exposes_official_and_experimental_implementations():
    assert EXTRACTOR_REGISTRY.names() == ("alphapose_full", "bytetrack_alphapose", "wham")
    with pytest.raises(RuntimeError, match="experimental"):
        build_extractor("wham", {})


def test_external_backend_adapters_have_no_personal_path_fallbacks():
    with pytest.raises(ValueError, match="repo_root is required"):
        AlphaPoseFullEstimator(AlphaPoseFullConfig())
    with pytest.raises(ValueError, match="repo_root is required"):
        ByteTrackTracker._resolve_repo_path(ByteTrackConfig().repo_root)
