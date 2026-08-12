from types import SimpleNamespace

import pytest

from src.engine.checkpoint_selection import resolve_selection_metric, selection_value_and_score


def test_auto_checkpoint_selection_uses_model_capability() -> None:
    capabilities = SimpleNamespace(preferred_validation_metric="val_loss")
    metric = resolve_selection_metric("auto", capabilities, has_validation=True)
    value, score = selection_value_and_score(metric, {"loss": 1.25, "top1": 0.75}, train_top1=0.8)

    assert metric == "val_loss"
    assert value == 1.25
    assert score == -1.25


def test_checkpoint_selection_falls_back_to_train_without_validation() -> None:
    capabilities = SimpleNamespace(preferred_validation_metric="val_loss")

    assert resolve_selection_metric("auto", capabilities, has_validation=False) == "train_top1"


def test_explicit_validation_metric_requires_validation_data() -> None:
    capabilities = SimpleNamespace(preferred_validation_metric="val_loss")

    with pytest.raises(ValueError, match="requires a non-empty validation split"):
        resolve_selection_metric("val_top1", capabilities, has_validation=False)
