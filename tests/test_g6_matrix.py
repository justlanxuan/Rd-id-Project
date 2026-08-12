from __future__ import annotations

from dataclasses import replace

import pytest

from tools.g6.matrix import build_required_cells, validate_required_cells


def test_g6_matrix_has_all_training_evaluation_seed_and_session_cells():
    cells = build_required_cells()

    assert len(cells) == 108
    assert sum(cell.job_type == "train" for cell in cells) == 42
    assert sum(cell.job_type == "evaluate" for cell in cells) == 66
    assert {cell.seed for cell in cells} == {0, 42, 123}
    custom_sessions = {cell.test_session for cell in cells if cell.dataset == "custom"}
    assert custom_sessions == {
        "20260211_171423",
        "20260211_171724",
        "20260211_172257",
        "20260211_172522",
    }


def test_g6_matrix_validator_rejects_duplicates_and_wrong_session():
    cells = build_required_cells()
    with pytest.raises(ValueError, match="duplicate"):
        validate_required_cells(cells + [cells[0]])

    custom_index = next(index for index, cell in enumerate(cells) if cell.dataset == "custom")
    invalid = list(cells)
    invalid[custom_index] = replace(invalid[custom_index], test_session="leaked_session")
    with pytest.raises(ValueError, match="wrong held-out session"):
        validate_required_cells(invalid)
