# Experiment Note: E5-A1-domain-and-orientation-screen
"""Run the preregistered E5 source-regime/orientation screen."""

from __future__ import annotations

from pathlib import Path

from tools.g12.train_orientation_matcher import main as train_main

ROOT = Path("/data/fzliang/reid-project")
E5 = ROOT / "g12/e5_egohumans_orientation"
EGO_ROOT = ROOT / "g12/e4_1_source_aligned/motionbert_alphapose_cache_v2"
TC_ROOT = ROOT / "g12/e4_1_source_aligned/motionbert_alphapose_cache_v2"
CUSTOM_CACHE = ROOT / "g12/e4_1_source_aligned/motionbert_alphapose_cache_v2"
EGO_GYRO = ROOT / "g10/e1_global_features/gyro_sidecar_egohumans"
TC_GYRO = ROOT / "g10/e1_global_features/gyro_sidecar_totalcapture"
CUSTOM_GYRO = ROOT / "g11/custom_complete_sidecars_v1"


def spec(dataset: str, csv: Path, root: Path, fps: int, gyro: Path, *, session: str | None = None) -> str:
    value = f"dataset={dataset};csv={csv};root={root};fps_hz={fps};gyro_sidecar_root={gyro}"
    return f"{value};session_filter={session}" if session else value


# E5 source-side runs must use the same extractor-derived cache as Custom/TC.
# The canonical 3D split remains a diagnostic control, not the primary result.
EGO_TRAIN = ROOT / "g12/e4_1_source_aligned/motionbert_alphapose_cache_v2/manifests/e5_ego_train.csv"
EGO_VAL = ROOT / "g12/e4_1_source_aligned/motionbert_alphapose_cache_v2/manifests/e5_ego_validation.csv"
EGO_TEST = ROOT / "g11/e3_raw_multiscale/manifests/egohumans_test_w0p8s.csv"
TC_TRAIN = ROOT / "g12/e4_1_source_aligned/motionbert_alphapose_cache_v2/manifests/tc_train.csv"
CUSTOM_VAL = CUSTOM_CACHE / "manifests/custom23_validation.csv"
CUSTOM_TEST = CUSTOM_CACHE / "manifests/custom23_test.csv"
CUSTOM_EVAL = CUSTOM_CACHE / "manifests/custom23_eval.csv"

EGO_TRAIN_SPEC = spec("egohumans_alphapose_e5", EGO_TRAIN, EGO_ROOT, 20, EGO_GYRO)
EGO_VAL_SPEC = spec("egohumans_alphapose_validation", EGO_VAL, EGO_ROOT, 20, EGO_GYRO)
# No source-aligned E5 test cache exists for the seven canonical test sessions;
# keep the canonical test as a labelled, non-promotional domain-shift diagnostic.
EGO_TEST_SPEC = spec("egohumans_canonical_test_diagnostic", EGO_TEST, ROOT / "egohumans/preprocessed/egohumans_realistic_hybrid_source", 20, EGO_GYRO)
TC_TRAIN_SPEC = spec("totalcapture_alphapose_e5", TC_TRAIN, TC_ROOT, 60, TC_GYRO)
CUSTOM_VAL_SPEC = spec("custom23_validation", CUSTOM_VAL, CUSTOM_CACHE, 30, CUSTOM_GYRO)
CUSTOM_TEST_SPEC = spec("custom23_test", CUSTOM_TEST, CUSTOM_CACHE, 30, CUSTOM_GYRO)
CUSTOM_CONTROLS = tuple(
    spec(name, CUSTOM_EVAL, CUSTOM_CACHE, 30, CUSTOM_GYRO, session=session)
    for name, session in (
        ("custom57", "20260211_171724"),
        ("custom22", "20260211_172257"),
        ("custom24", "20260211_172522"),
    )
)


def _args(
    regime: str,
    variant: str,
    profile: str,
    seed: int,
    *,
    orientation_mode: str = "3d_heading",
    output_subdir: str = "screen_fully_aligned",
) -> list[str]:
    if regime == "eh_only":
        train_specs = [EGO_TRAIN_SPEC]
    elif regime == "tc_only":
        train_specs = [TC_TRAIN_SPEC]
    elif regime == "tc_eh_balanced":
        train_specs = [TC_TRAIN_SPEC, EGO_TRAIN_SPEC]
    else:
        raise ValueError(regime)
    output = E5 / output_subdir / f"{regime}_{variant}_{profile}_seed{seed}"
    args = [
        "--variant", variant,
        "--orientation-mode", orientation_mode,
        "--orientation-profile", profile,
        "--selection-domain", "custom23_validation",
        "--selection-stratum", "high",
        "--turning-threshold", str(19.0 / 48.0),
        "--target-len", "24",
        "--window-seconds", "0.8",
        "--hidden", "96",
        "--embedding-dim", "64",
        "--batch-size", "32",
        "--epochs", "3",
        "--steps-per-epoch", "50",
        "--lr", "0.002",
        "--temperature", "0.1",
        "--aux-turning-weight", "0.05" if variant != "baseline" else "0.0",
        "--train-spec", train_specs[0],
        "--eval-spec", EGO_VAL_SPEC,
        "--eval-spec", CUSTOM_VAL_SPEC,
        "--test-spec", EGO_TEST_SPEC,
        "--test-spec", CUSTOM_TEST_SPEC,
        "--test-spec", CUSTOM_CONTROLS[0],
        "--test-spec", CUSTOM_CONTROLS[1],
        "--test-spec", CUSTOM_CONTROLS[2],
        "--output", str(output),
        "--seed", str(seed),
        "--device", "cuda:6",
    ]
    if len(train_specs) == 2:
        args.extend(("--train-spec", train_specs[1]))
    return args


def main() -> int:
    # Screen first; only candidates that pass the validation gate are promoted
    # to the 5-seed frozen confirmation script.
    for regime in ("eh_only", "tc_only", "tc_eh_balanced"):
        for variant, profile in (("baseline", "full"), ("turning_cross", "full"), ("turning_cross", "rate")):
            for seed in range(3):
                train_main(_args(regime, variant, profile, seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
