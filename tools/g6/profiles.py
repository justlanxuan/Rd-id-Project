"""Data/config profiles for the official benchmark and controlled ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    base_configs: dict[str, Path]
    custom_base_configs: dict[int, Path]
    source_roots: dict[str, Path]
    custom_root: Path


PROFILES = {
    "g6": BenchmarkProfile(
        name="g6",
        base_configs={
            "totalcapture": REPO_ROOT / "configs/g6/totalcapture_source.yaml",
            "egohumans": REPO_ROOT / "configs/g6/egohumans_source.yaml",
        },
        custom_base_configs={
            fold_id: REPO_ROOT / f"configs/g6/custom_direct_fold{fold_id}.yaml"
            for fold_id in range(1, 5)
        },
        source_roots={
            "totalcapture": Path("/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source"),
            "egohumans": Path("/data/fzliang/reid-project/egohumans/preprocessed/g6_egohumans_source"),
        },
        custom_root=Path(
            "/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_session_out_rawcsv7d_swapsess"
        ),
    ),
    "stride24": BenchmarkProfile(
        name="stride24",
        base_configs={
            "totalcapture": REPO_ROOT / "configs/stride24/totalcapture_source.yaml",
            "egohumans": REPO_ROOT / "configs/stride24/egohumans_source.yaml",
        },
        custom_base_configs={
            fold_id: REPO_ROOT / f"configs/stride24/custom_direct_fold{fold_id}.yaml"
            for fold_id in range(1, 5)
        },
        source_roots={
            "totalcapture": Path("/data/fzliang/reid-project/totalcapture/preprocessed/stride24_w24"),
            "egohumans": Path("/data/fzliang/reid-project/egohumans/preprocessed/stride24_w24"),
        },
        custom_root=Path(
            "/data/fzliang/reid-project/custom/preprocessed/stride24_w24_rawcsv7d_swapsess"
        ),
    ),
}


def get_profile(name: str) -> BenchmarkProfile:
    key = str(name).strip().lower()
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark profile {name!r}; available={sorted(PROFILES)}") from exc
