from __future__ import annotations

import pytest

from preprocess.datasets.custom_plus import run_preprocess


def test_custom_plus_imports_but_fails_with_actionable_experimental_error():
    with pytest.raises(RuntimeError, match="CustomPlusAdapter"):
        run_preprocess(None)
