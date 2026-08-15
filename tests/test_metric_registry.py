from __future__ import annotations

import pytest

from src.metrics import METRIC_REGISTRY, FrameAccEvaluator, GroupTestEvaluator, build_metric


def test_metric_registry_exposes_independent_metric_types():
    assert METRIC_REGISTRY.names() == ("frame_acc", "group_test")
    assert isinstance(build_metric("frame_acc"), FrameAccEvaluator)
    assert isinstance(build_metric("group_test", group_sizes=[2]), GroupTestEvaluator)


def test_metric_registry_rejects_unknown_metric():
    with pytest.raises(KeyError, match="Unknown metric"):
        build_metric("accuracy_that_does_not_exist")
