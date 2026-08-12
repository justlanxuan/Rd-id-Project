from __future__ import annotations

from tools.g6.compare_results import compare_results


def _result(value):
    overall = {}
    for key in (
        "zero_shot.egohumans",
        "zero_shot.totalcapture",
        "finetune.egohumans",
        "finetune.totalcapture",
        "direct.none",
    ):
        condition, source = key.split(".")
        overall[key] = {
            "condition": condition,
            "source": None if source == "none" else source,
            "macro_session": {"mean": value},
        }
    return {
        "protocol_hash": "a" * 64,
        "source": {
            "totalcapture": {"mean": value},
            "egohumans": {"mean": value},
        },
        "custom_overall": overall,
    }


def test_compare_results_reports_variant_minus_baseline():
    text = compare_results(
        _result(0.5),
        _result(0.6),
        baseline_label="old",
        variant_label="new",
    )
    assert "Delta is variant − baseline" in text
    assert text.count("+10.00 pp") == 7
