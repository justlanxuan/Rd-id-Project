# Experiment Note: E1-orientation-inventory-validation
"""Validate the generated E1 inventory contract without touching source data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPECTED_CLASSES = {"direct", "derived", "proxy", "missing"}


def main(path: str) -> int:
    inventory_path = Path(path).expanduser().resolve()
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "g12.orientation_inventory.v1"
    assert payload["read_only"] is True
    assert set(payload["orientation_classes"]) == EXPECTED_CLASSES
    expected_hash = payload.pop("manifest_sha256")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    payload["manifest_sha256"] = expected_hash

    records = payload["records"]
    assert records, "inventory contains no source records"
    for record in records:
        assert record["orientation_class"] in EXPECTED_CLASSES
        assert record["status"]
        assert "coordinate_frame" in record
        assert record["time_fields"]
        assert record["provenance"]
        assert record["fingerprint"]["fingerprint_sha256"]
        for sample in record.get("samples", []):
            assert sample["file"]["sha256"]
            if "finite" in sample:
                assert sample["finite"] is True
            for value in sample.values():
                if isinstance(value, dict) and "finite" in value:
                    assert value["finite"] is True

    by_id = {record["source_id"]: record for record in records}
    tc = by_id["totalcapture_vicon_orientation"]
    assert tc["orientation_class"] == "direct"
    assert tc["archive_orientation_member_count"] > 0
    assert tc["samples"][0]["format"] == "quaternion_wxyz"
    assert 0.999 < tc["samples"][0]["quaternion_norm_min"] < 1.001
    assert 0.999 < tc["samples"][0]["quaternion_norm_max"] < 1.001

    assert by_id["totalcapture_smplx_root_orientation"]["orientation_class"] == "direct"
    assert by_id["egohumans_fitted_smpl_global_orientation"]["orientation_class"] == "direct"
    assert by_id["fzliang_totalcapture_canonical"]["orientation_class"] == "derived"
    assert by_id["fzliang_egohumans_canonical"]["orientation_class"] == "proxy"
    assert by_id["fzliang_custom_canonical"]["orientation_class"] == "missing"
    print(f"Validated {len(records)} orientation records; manifest hash={expected_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/data/fzliang/reid-project/g12/e1_inventory/orientation_inventory.json"))
