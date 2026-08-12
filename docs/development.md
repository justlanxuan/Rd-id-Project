# Development Guide

## Code boundaries

- `run_pipeline.py` is the only public workflow entrypoint.
- `src/workflow/` owns stage construction and ordering contracts.
- `src/config/` owns defaults, legacy YAML normalization, and path resolution.
- `preprocess/` owns raw dataset conversion, extraction dispatch, canonical
  packing/slicing, adapters, and prepared-cache validation.
- `src/datasets/` reads canonical NPZ/CSV artifacts; it does not know raw
  dataset layouts.
- `src/models/` owns model construction, capabilities, and checkpoints.
- `src/metrics/` owns metric construction and exact aggregation semantics.
- `src/engine/` owns training/evaluation loops against those interfaces.
- `tools/g6/` owns reproducible experiment matrices and execution tooling.
- `experiments/` contains plans, reports, and one-off research artifacts.

Production modules must not import from `experiments/`.

## Compatibility policy

The stable external contract is:

```bash
python run_pipeline.py --config CONFIG [--stages preprocess,train,test]
```

`prepare` and `evaluate` are temporary deprecated stage aliases. Deleted
`src.pipeline`, `src.pipelines`, and `src.preprocess` paths are not public API.
If a future rename must remain compatible, use a thin, tested shim with an
explicit removal condition.

## Adding domains

### Dataset

Implement and register a `DatasetAdapter` under `preprocess/adapters/`. Emit
the canonical schema and keep using `WindowAlignmentDataset` unless the
canonical reading semantics genuinely differ.

### Extractor

Implement `VideoSkeletonExtractor`, dependency checks, cache validation, and
provenance. A production claim requires a real forced short-video test with
non-empty output.

### Model

Implement the model input/output contract and capabilities, then register its
builder under `src/models/`. Add a model-owned checkpoint adapter where needed
and test forward, backward, save/load, and every advertised evaluator.

### Metric

Implement `EvaluationMetric` under `src/metrics/`. Define raw counts,
exclusions, degenerate cases, and aggregation before registering it.

## Training code

- metadata conversion: `src/engine/batch.py`
- augmentation: `src/engine/augmentation.py`
- losses: `src/engine/losses.py`
- fitted statistics: `src/engine/stats.py`
- validation loop: `src/engine/validation.py`
- construction shared by train/test: `src/engine/common.py`

Keep `train.py` and `evaluate.py` focused on orchestration. Model-name branches
belong in model capabilities/adapters, not the engines.

## Checks

Use the project environment rather than an unrelated base Python:

```bash
python -m compileall -q run_pipeline.py preprocess src tools/g6
python -m pytest -q
python -m ruff check run_pipeline.py preprocess src tools/g6 tests
git diff --check
```

For data or extractor changes, also run the corresponding real adapter or
short-video smoke; toy tests alone do not establish real backend compatibility.
