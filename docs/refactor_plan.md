# Refactor Plan for Official Open-Source Readiness

This document records the low-risk refactor plan for making the repository easier
to understand, extend, and maintain in multi-developer workflows without changing
the current training and evaluation behavior.

## Goals

- Keep the official pipeline behavior unchanged.
- Make the official entrypoints obvious.
- Separate production code from experiment artifacts.
- Make dataset, preprocessing, training, and model boundaries explicit.
- Preserve legacy import paths while introducing clearer module names.

## Target Boundaries

```text
src/pipeline.py        # Official workflow entrypoint.
src/config/            # Defaults, config loading, path resolution.
src/preprocess/        # Raw dataset -> standardized NPZ/CSV.
src/datasets/          # Standardized NPZ/CSV -> PyTorch Dataset/DataLoader.
src/engine/            # Train/evaluate orchestration and engine helpers.
src/modules/           # Models, encoders, matchers, losses.
src/utils/             # Small shared utilities.
experiments/           # One-off research scripts, reports, and ablations.
configs/official/      # Recommended reproducible configs.
configs/examples/      # Small smoke-test configs.
configs/experiments/   # Sweep and ablation configs.
```

## Minimal Refactor Steps

1. Document the official structure and keep README aligned with the actual tree.
2. Split `src/datasets/alignment_dataset.py` into focused dataset modules:
   - `alignment.py`: `WindowAlignmentDataset`
   - `transforms.py`: IMU filtering and legacy single-sensor conversion
   - `samplers.py`: dataset-aware batch samplers
   - keep `alignment_dataset.py` as a compatibility shim
3. Extract focused helpers from `src/engine/train.py`:
   - `batch.py`: moving batches to device and converting metadata to labels
   - `losses.py`: training-specific contrastive and pair losses
   - `stats.py`: IMU and hybrid encoder statistics
4. Keep public behavior stable:
   - no config semantics changes
   - no checkpoint key changes
   - no default stage changes
   - no removal of old import paths in this pass
5. Add smoke checks after each refactor:
   - Python bytecode compile for touched modules
   - import checks for old and new dataset paths
   - config load checks for representative configs

## Follow-Up Cleanup

These are intentionally not bundled into the first code refactor because they
touch many files and should be reviewed separately.

- Move official configs into `configs/official/` and smoke configs into
  `configs/examples/`.
- Move `_tmp_*.yaml`, seed sweeps, and ablation configs into
  `configs/experiments/` or keep them untracked.
- Remove tracked `__pycache__` and generated files.
- Add `pyproject.toml`, formatter/linter settings, and a small `tests/` suite.
- Ensure experiment scripts do not become dependencies of `src/`.
