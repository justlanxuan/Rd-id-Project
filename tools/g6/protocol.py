"""Build the immutable protocol identity used by every formal G6 run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data_manifest import sha256_file
from .git_snapshot import require_clean_git_snapshot
from .matrix import build_required_cells
from .profiles import get_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_FILE = REPO_ROOT / "environment.yml"


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_protocol_record(
    protocol_document: str | Path,
    data_manifest_index: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    profile_name: str = "g6",
) -> dict[str, Any]:
    document = Path(protocol_document).expanduser().resolve()
    text = document.read_text(encoding="utf-8")
    if "状态：`locked`" not in text:
        raise ValueError("Protocol document must have status `locked` before hashing.")

    index_path = Path(data_manifest_index).expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    data_hashes = {
        f"{row['dataset']}.fold{row.get('fold_id') if row.get('fold_id') is not None else 'none'}": str(row["manifest_hash"])
        for row in index.get("manifests", [])
    }
    if len(data_hashes) != 6 or any(not value for value in data_hashes.values()):
        raise ValueError("Protocol requires exactly six non-empty G6 data manifest hashes.")

    profile = get_profile(profile_name)
    config_paths = {
        **profile.base_configs,
        **{f"custom_fold{key}": value for key, value in profile.custom_base_configs.items()},
    }
    base_config_hashes = {
        str(name): sha256_file(path.resolve())
        for name, path in sorted(config_paths.items(), key=lambda item: str(item[0]))
    }
    cells = [cell.to_dict() for cell in build_required_cells()]
    git_commit = require_clean_git_snapshot(repo_root)
    components = {
        "profile": profile.name,
        "git_commit": git_commit,
        "environment_sha256": sha256_file(ENVIRONMENT_FILE),
        "protocol_document_sha256": sha256_file(document),
        "data_manifest_hashes": data_hashes,
        "base_config_sha256": base_config_hashes,
        "required_cells_sha256": _sha256_json(cells),
    }
    return {
        "schema_version": "1.0",
        "status": "locked",
        "profile": profile.name,
        "git_commit": git_commit,
        "protocol_hash": _sha256_json(components),
        "components": components,
        "summary": {"training": 42, "evaluation": 66, "total": 108},
    }
