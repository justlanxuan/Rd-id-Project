from __future__ import annotations

import json

import pytest

from tools.g6.protocol import build_protocol_record


def _write_data_index(path):
    rows = [
        {"dataset": "totalcapture", "fold_id": None, "manifest_hash": "tc"},
        {"dataset": "egohumans", "fold_id": None, "manifest_hash": "ego"},
    ]
    rows.extend(
        {"dataset": "custom", "fold_id": fold, "manifest_hash": f"custom-{fold}"}
        for fold in range(1, 5)
    )
    path.write_text(json.dumps({"manifests": rows}), encoding="utf-8")


def test_protocol_record_requires_explicit_lock_and_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.g6.protocol.require_clean_git_snapshot",
        lambda _repo_root: "0123456789abcdef0123456789abcdef01234567",
    )
    document = tmp_path / "protocol-lock.md"
    index = tmp_path / "index.json"
    _write_data_index(index)
    document.write_text("状态：`awaiting_human_confirmation`\n", encoding="utf-8")
    with pytest.raises(ValueError, match="status `locked`"):
        build_protocol_record(document, index)

    document.write_text("状态：`locked`\nprotocol body\n", encoding="utf-8")
    first = build_protocol_record(document, index)
    second = build_protocol_record(document, index)
    assert first == second
    assert len(first["protocol_hash"]) == 64
    assert first["git_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert first["summary"] == {"training": 42, "evaluation": 66, "total": 108}
