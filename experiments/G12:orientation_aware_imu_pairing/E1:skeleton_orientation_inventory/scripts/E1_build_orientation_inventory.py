# Experiment Note: E1-orientation-inventory
"""Build a read-only inventory of available skeleton orientation signals.

The scanner intentionally inspects representative files from large datasets
and records counts, byte totals, field schemas, finite checks and stable
content fingerprints. It never extracts archives, rewrites source data, or
modifies the project's canonical preprocessing schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_LYXIE = Path("/data/lyxie")
DEFAULT_FZLIANG = Path("/data/fzliang")
DEFAULT_OUTPUT = Path("/data/fzliang/reid-project/g12/e1_inventory")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fingerprint(paths: Iterable[Path], root: Path, limit: int) -> dict[str, Any]:
    paths = sorted({path.resolve() for path in paths if path.is_file()})
    total_bytes = sum(path.stat().st_size for path in paths)
    if len(paths) <= limit:
        selected = paths
        scope = "all_files"
    else:
        indices = sorted({0, len(paths) // 2, len(paths) - 1})
        selected = [paths[index] for index in indices]
        scope = "first_middle_last_samples"
    records = []
    for path in selected:
        records.append(
            {
                "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "hash_scope": scope,
        "sample_limit": limit,
        "samples": records,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _sample_paths(paths: Iterable[Path], limit: int) -> list[Path]:
    paths = sorted({path.resolve() for path in paths if path.is_file()})
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[int(index)] for index in sorted(set(indices))]


def _finite_summary(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if np.issubdtype(array.dtype, np.number):
        summary["finite"] = bool(np.isfinite(array).all())
        if array.size:
            summary["min"] = float(np.nanmin(array))
            summary["max"] = float(np.nanmax(array))
    return summary


def _record_path(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _scan_tc_vicon(lyxie_root: Path, sample_limit: int, hash_limit: int) -> dict[str, Any]:
    raw_root = lyxie_root / "ReID_imu_generation/data/raw/totalcapture"
    ori_paths = sorted(raw_root.rglob("gt_skel_gbl_ori.txt")) if raw_root.is_dir() else []
    pos_paths = sorted(raw_root.rglob("gt_skel_gbl_pos.txt")) if raw_root.is_dir() else []
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in _sample_paths(ori_paths, sample_limit):
        try:
            with path.open(errors="replace") as handle:
                header = handle.readline().strip().split("\t")
                rows = [np.fromstring(line, sep="\t") for line in handle if line.strip()]
            values = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float64)
            per_joint = values.shape[1] // len(header) if header and values.size else 0
            reshaped = values.reshape(values.shape[0], len(header), per_joint) if per_joint == 4 else None
            norms = np.linalg.norm(reshaped, axis=-1) if reshaped is not None else np.empty(0)
            samples.append(
                {
                    "file": _record_path(path, lyxie_root),
                    "header_joint_count": len(header),
                    "header_joints": header,
                    "value_shape": list(values.shape),
                    "format": "quaternion_wxyz" if per_joint == 4 else "unknown",
                    "finite": bool(np.isfinite(values).all()),
                    "quaternion_norm_min": float(norms.min()) if norms.size else None,
                    "quaternion_norm_max": float(norms.max()) if norms.size else None,
                    "timestamp_field": "implicit row index; source sampling must be resolved from protocol",
                    "coordinate_frame": "global (declared by gbl filename; convention still requires audit)",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive inventory path
            failures.append({"path": str(path), "error": repr(exc)})
    archive_roots = [lyxie_root / "ReID/Data/totalcapture", lyxie_root / "ReID/Data/Experiment/Original-4-Person"]
    archives = sorted({path.resolve() for archive_root in archive_roots if archive_root.is_dir() for path in archive_root.glob("*_vicon_pos_ori.tar.gz")})
    archive_members = []
    archive_orientation_member_count = 0
    archive_position_member_count = 0
    for path in archives:
        try:
            with tarfile.open(path, "r:gz") as archive:
                names = archive.getnames()
            ori_members = [name for name in names if name.endswith("gt_skel_gbl_ori.txt")]
            pos_members = [name for name in names if name.endswith("gt_skel_gbl_pos.txt")]
            archive_orientation_member_count += len(ori_members)
            archive_position_member_count += len(pos_members)
            archive_members.append(
                {
                    "file": _record_path(path, lyxie_root),
                    "orientation_members": len(ori_members),
                    "position_members": len(pos_members),
                    "member_sample": ori_members[:5],
                    "extracted": False,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive archive path
            failures.append({"path": str(path), "error": repr(exc)})
    return {
        "source_id": "totalcapture_vicon_orientation",
        "orientation_class": "direct",
        "status": "candidate",
        "evidence": ["gt_skel_gbl_ori.txt", "gt_skel_gbl_pos.txt", "21-joint quaternion header"],
        "root": str(lyxie_root),
        "discovered_orientation_files": len(ori_paths),
        "discovered_position_files": len(pos_paths),
        "archive_count": len(archives),
        "archive_orientation_member_count": archive_orientation_member_count,
        "archive_position_member_count": archive_position_member_count,
        "archive_member_summary": archive_members,
        "samples": samples,
        "failures": failures,
        "coordinate_frame": "global, exact convention pending",
        "time_fields": ["implicit row index"],
        "provenance": "TotalCapture Vicon optical motion capture; raw orientation files are not currently passed by the canonical adapter",
        "fingerprint": _stable_fingerprint(ori_paths + pos_paths, lyxie_root, hash_limit),
    }


def _scan_tc_smplx(lyxie_root: Path, sample_limit: int, hash_limit: int) -> dict[str, Any]:
    root = lyxie_root / "ReID_imu_generation/data/processed/totalcapture_test"
    paths = sorted(root.glob("*/*_smplx.npz")) if root.is_dir() else []
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in _sample_paths(paths, sample_limit):
        try:
            with np.load(path, allow_pickle=True) as data:
                keys = list(data.files)
                root_orient = np.asarray(data["root_orient"]) if "root_orient" in data.files else None
                pose_body = np.asarray(data["pose_body"]) if "pose_body" in data.files else None
                samples.append(
                    {
                        "file": _record_path(path, lyxie_root),
                        "keys": keys,
                        "root_orient": _finite_summary(root_orient) if root_orient is not None else None,
                        "pose_body": _finite_summary(pose_body) if pose_body is not None else None,
                        "mocap_frame_rate": float(np.asarray(data["mocap_frame_rate"]).item()) if "mocap_frame_rate" in data.files else None,
                        "coordinate_frame": "SMPL-X global frame; exact relation to Vicon pending",
                        "timestamp_field": "implicit mocap frame index",
                    }
                )
        except Exception as exc:
            failures.append({"path": str(path), "error": repr(exc)})
    return {
        "source_id": "totalcapture_smplx_root_orientation",
        "orientation_class": "direct",
        "status": "candidate",
        "evidence": ["root_orient", "pose_body", "trans"],
        "root": str(lyxie_root),
        "discovered_files": len(paths),
        "samples": samples,
        "failures": failures,
        "coordinate_frame": "SMPL-X global frame, convention pending",
        "time_fields": ["implicit mocap frame index", "mocap_frame_rate when present"],
        "provenance": "TotalCapture SMPL-X processed artifact; root_orient is axis-angle",
        "fingerprint": _stable_fingerprint(paths, lyxie_root, hash_limit),
    }


def _scan_ego_smpl(lyxie_root: Path, sample_limit: int, hash_limit: int) -> dict[str, Any]:
    root = lyxie_root / "ReID/Data/egohumans/data"
    paths = sorted(root.glob("*/*/processed_data/smpl/*.npy")) if root.is_dir() else []
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in _sample_paths(paths, sample_limit):
        try:
            payload = np.load(path, allow_pickle=True).item()
            people = sorted(payload) if isinstance(payload, dict) else []
            person = payload[people[0]] if people else None
            orientation = person.get("global_orient") if isinstance(person, dict) else None
            samples.append(
                {
                    "file": _record_path(path, lyxie_root),
                    "person_count": len(people),
                    "person_ids": people,
                    "global_orient": _finite_summary(np.asarray(orientation)) if orientation is not None else None,
                    "available_keys": sorted(person) if isinstance(person, dict) else [],
                    "coordinate_frame": "SMPL fitted global frame; camera/world convention pending",
                    "timestamp_field": "filename frame index",
                }
            )
        except Exception as exc:
            failures.append({"path": str(path), "error": repr(exc)})
    return {
        "source_id": "egohumans_fitted_smpl_global_orientation",
        "orientation_class": "direct",
        "status": "candidate_estimated",
        "evidence": ["global_orient", "body_pose", "transl", "joints"],
        "root": str(lyxie_root),
        "discovered_files": len(paths),
        "samples": samples,
        "failures": failures,
        "coordinate_frame": "SMPL fitted global frame, convention pending",
        "time_fields": ["filename frame index"],
        "provenance": "EgoHumans fitted SMPL estimate; not optical ground truth",
        "fingerprint": _stable_fingerprint(paths, lyxie_root, hash_limit),
    }


def _scan_ego_pose3d(lyxie_root: Path, sample_limit: int, hash_limit: int) -> list[dict[str, Any]]:
    specs = [
        ("egohumans_extracted_pose3d", lyxie_root / "ReID/Data/egohumans/extracted_data", "*.npy", "pose3d"),
        ("egohumans_fit_pose3d", lyxie_root / "ReID/Data/egohumans/data", "*/*/processed_data/fit_poses3d/*.npy", "fit_poses3d"),
        ("egohumans_refine_pose3d", lyxie_root / "ReID/Data/egohumans/data", "*/*/processed_data/refine_poses3d/*.npy", "refine_poses3d"),
    ]
    records = []
    for source_id, root, pattern, field_name in specs:
        paths = sorted(root.glob(pattern)) if root.is_dir() else []
        samples: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for path in _sample_paths(paths, sample_limit):
            try:
                payload = np.load(path, allow_pickle=True).item()
                if field_name == "pose3d":
                    array = np.asarray(payload["pose3d"])
                    people = None
                else:
                    people = sorted(payload) if isinstance(payload, dict) else []
                    array = np.asarray(payload[people[0]]) if people else np.empty(0)
                samples.append(
                    {
                        "file": _record_path(path, lyxie_root),
                        "field": field_name,
                        "people": people,
                        "array": _finite_summary(array),
                        "coordinate_frame": "3D pose frame pending convention audit",
                        "timestamp_field": "implicit row/frame index",
                    }
                )
            except Exception as exc:
                failures.append({"path": str(path), "error": repr(exc)})
        records.append(
            {
                "source_id": source_id,
                "orientation_class": "derived",
                "status": "derived_only",
                "evidence": [field_name],
                "root": str(lyxie_root),
                "discovered_files": len(paths),
                "samples": samples,
                "failures": failures,
                "coordinate_frame": "unknown 3D pose frame",
                "time_fields": ["implicit row/frame index"],
                "provenance": "EgoHumans 3D pose; heading requires pelvis/hip/shoulder derivation and sign validation",
                "fingerprint": _stable_fingerprint(paths, lyxie_root, hash_limit),
            }
        )
    return records


def _scan_canonical(root: Path, dataset: str, orientation_class: str, status: str, sample_limit: int, hash_limit: int) -> dict[str, Any]:
    prepared = root / "reid-project" / dataset / "preprocessed"
    patterns = sorted(prepared.glob("*/sequences/*.npz")) if prepared.is_dir() else []
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in _sample_paths(patterns, sample_limit):
        try:
            with np.load(path, allow_pickle=True) as data:
                keys = list(data.files)
                skeleton_key = "gt_skeleton_meters" if "gt_skeleton_meters" in keys else ("gt_skeleton" if "gt_skeleton" in keys else "skeleton")
                skeleton = np.asarray(data[skeleton_key]) if skeleton_key in keys else np.empty(0)
                orientation_keys = [key for key in keys if any(token in key.lower() for token in ("orient", "rotation", "quaternion", "yaw", "heading"))]
                samples.append(
                    {
                        "file": _record_path(path, root),
                        "dataset": str(np.asarray(data["dataset"]).item()) if "dataset" in keys else dataset,
                        "keys": keys,
                        "skeleton_field": skeleton_key,
                        "skeleton": _finite_summary(skeleton),
                        "orientation_keys": orientation_keys,
                        "coordinate_frame": "canonical frame; source semantics inherited and must be resolved",
                        "timestamp_field": "timestamps_s when present, otherwise frame_ids",
                    }
                )
        except Exception as exc:
            failures.append({"path": str(path), "error": repr(exc)})
    notes = {
        "totalcapture": "3D gt_skeleton_meters permits derived heading, but raw Vicon orientation is not present in canonical files",
        "egohumans": "gt_skeleton is 2D xy plus visibility; no reliable world orientation",
        "custom": "skeleton/extract_skeleton is 2D; current corrected cache is orientation-missing for world yaw",
    }
    return {
        "source_id": f"fzliang_{dataset}_canonical",
        "orientation_class": orientation_class,
        "status": status,
        "evidence": ["canonical sequence NPZ"],
        "root": str(root),
        "discovered_files": len(patterns),
        "samples": samples,
        "failures": failures,
        "coordinate_frame": "canonical frame, pending source-specific audit",
        "time_fields": ["timestamps_s when present", "frame_ids"],
        "provenance": notes[dataset],
        "fingerprint": _stable_fingerprint(patterns, root, hash_limit),
    }


def _manifest(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "g12.orientation_inventory.v1",
        "created_by": "E1_build_orientation_inventory.py",
        "read_only": True,
        "source_roots": {"lyxie": str(args.lyxie_root), "fzliang": str(args.fzliang_root)},
        "sample_limit": args.sample_limit,
        "hash_limit": args.hash_limit,
        "orientation_classes": ["direct", "derived", "proxy", "missing"],
        "records": records,
    }


def _write_markdown(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# G12 E1 Orientation Inventory",
        "",
        f"- Schema: `{manifest['schema_version']}`",
        f"- Read-only: `{manifest['read_only']}`",
        f"- Sample limit: `{manifest['sample_limit']}`; hash limit: `{manifest['hash_limit']}`",
        "",
        "## Summary",
        "",
        "| Source | Class | Status | Files | Samples | Failures | Provenance |\n|---|---|---|---:|---:|---:|---|",
    ]
    for record in manifest["records"]:
        if record["source_id"] == "totalcapture_vicon_orientation":
            file_count = (
                f"{record['discovered_orientation_files']} raw + {record['archive_count']} archives "
                f"({record['archive_orientation_member_count']} members)"
            )
        else:
            file_count = str(record.get("discovered_files", record.get("discovered_orientation_files", 0)))
        lines.append(
            f"| `{record['source_id']}` | `{record['orientation_class']}` | `{record['status']}` | {file_count} | "
            f"{len(record.get('samples', []))} | {len(record.get('failures', []))} | {record['provenance']} |"
        )
    lines.extend(["", "## Source details", ""])
    for record in manifest["records"]:
        lines.extend(
            [
                f"### `{record['source_id']}`",
                "",
                f"- Orientation class: `{record['orientation_class']}`",
                f"- Status: `{record['status']}`",
                f"- Coordinate frame: {record['coordinate_frame']}",
                f"- Time fields: {', '.join(record['time_fields'])}",
                f"- Provenance: {record['provenance']}",
                f"- Fingerprint: `{record['fingerprint']['fingerprint_sha256']}` ({record['fingerprint']['hash_scope']})",
                "",
            ]
        )
        if record.get("failures"):
            lines.append("Failures:")
            for failure in record["failures"]:
                lines.append(f"- `{failure['path']}`: `{failure['error']}`")
            lines.append("")
        lines.append("Representative samples:")
        for sample in record.get("samples", []):
            lines.append(f"- `{sample['file']['relative_path']}`: `{json.dumps(sample, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the G12 E1 skeleton orientation inventory")
    parser.add_argument("--lyxie-root", type=Path, default=DEFAULT_LYXIE)
    parser.add_argument("--fzliang-root", type=Path, default=DEFAULT_FZLIANG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--hash-limit", type=int, default=50)
    args = parser.parse_args()
    args.lyxie_root = args.lyxie_root.expanduser().resolve()
    args.fzliang_root = args.fzliang_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = [
        _scan_tc_vicon(args.lyxie_root, args.sample_limit, args.hash_limit),
        _scan_tc_smplx(args.lyxie_root, args.sample_limit, args.hash_limit),
        _scan_ego_smpl(args.lyxie_root, args.sample_limit, args.hash_limit),
        *_scan_ego_pose3d(args.lyxie_root, args.sample_limit, args.hash_limit),
        _scan_canonical(args.fzliang_root, "totalcapture", "derived", "derived_only", args.sample_limit, args.hash_limit),
        _scan_canonical(args.fzliang_root, "egohumans", "proxy", "orientation_missing", args.sample_limit, args.hash_limit),
        _scan_canonical(args.fzliang_root, "custom", "missing", "orientation_missing", args.sample_limit, args.hash_limit),
    ]
    manifest = _manifest(records, args)
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    json_path = args.output_dir / "orientation_inventory.json"
    markdown_path = args.output_dir / "orientation_inventory.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(manifest, markdown_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
