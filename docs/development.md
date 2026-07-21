# Development Guide

This repository keeps production code, reusable data preparation, and research
experiments separate.

## Code Boundaries

- `src/pipeline.py` is the official workflow entrypoint.
- `src/config/` owns defaults, legacy YAML normalization, and path resolution.
- `src/preprocess/` converts raw dataset sources into standardized NPZ/CSV.
- `src/datasets/` converts standardized NPZ/CSV into PyTorch datasets and
  dataset-aware samplers.
- `src/engine/` owns training/evaluation orchestration and engine helpers.
- `src/modules/` owns model, encoder, matcher, and reusable loss components.
- `experiments/` is for one-off scripts, reports, and ablations. Production
  code under `src/` should not import from `experiments/`.

## Compatibility Policy

Small refactors should preserve existing public entrypoints. When a module is
renamed, leave a compatibility shim for at least one development cycle. For
example, `src.datasets.alignment_dataset` still re-exports
`WindowAlignmentDataset` after the official implementation moved to
`src.datasets.alignment`.

## Adding a Dataset

1. Add dataset-specific raw conversion under `src/preprocess/datasets/`.
2. Emit the standardized NPZ schema and window CSVs described in
   `docs/data_format.md`.
3. Reuse `WindowAlignmentDataset` unless the standardized schema itself changes.
4. Add or update a smoke config under `configs/examples/` or an official config
   under `configs/official/` once those config directories are introduced.

## Adding Training Logic

- Put metadata conversion in `src/engine/batch.py`.
- Put input augmentation in `src/engine/augmentation.py`.
- Put training-only loss composition in `src/engine/losses.py`.
- Put statistics fitting in `src/engine/stats.py`.
- Put validation-loop helpers in `src/engine/validation.py`.
- Keep `src/engine/train.py` focused on config loading, construction, and the
  main training loop.

## Checks

Use the project Python environment, not the base conda environment. On this
machine, `mobind_repro` currently has the runtime dependencies needed for import
smoke checks:

```bash
/home/fzliang/miniconda3/envs/mobind_repro/bin/python -m compileall -q src
/home/fzliang/miniconda3/envs/mobind_repro/bin/python - <<'PY'
from src.config import load_cfg
from src.datasets import WindowAlignmentDataset
print(load_cfg("configs/totalcapture_vicon_test.yaml").PREPROCESS.DATASET)
print(WindowAlignmentDataset)
PY
```

After installing development dependencies, run:

```bash
python -m pytest
ruff check src tests
```
