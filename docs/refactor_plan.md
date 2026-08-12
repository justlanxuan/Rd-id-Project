# Production Refactor Plan

The authoritative refactor and experiment plan is:

```text
experiments/G6:official_refactor_and_three_dataset_benchmark/plan.md
```

This short document records the stable architectural decisions that should
survive individual experiments.

## Public contract

- one root CLI: `run_pipeline.py`;
- three public stages: `preprocess`, `train`, `test`;
- one configuration loader and path resolver;
- one canonical sequence/window format;
- independent registries for dataset adapters, extractors, models, metrics,
  and workflow stages;
- immutable run records and machine-generated aggregation.

## Dependency direction

```text
DatasetAdapter -> canonical artifact -> WindowAlignmentDataset
Extractor      -> canonical skeleton artifact
Model          -> ModelOutput/capabilities
Metric         -> raw counts and aggregate record
Workflow       -> stage orchestration only
```

No component may reintroduce a universal factory spanning these domains.

## Compatibility

Compatibility is preserved at deliberate public boundaries and through
versioned checkpoint/data adapters. Removed internal migration paths are not
kept alive by empty aliases. Legacy artifacts must either pass an explicit
migration test or fail with an actionable error.

## Completion gates

- no stale imports or documented commands on the official surface;
- full tests, targeted lint, compile/import, CLI and config checks pass;
- three real data adapters and one real forced extractor smoke per dataset;
- protocol/data/config/checkpoint hashes bind every formal result;
- all source, zero-shot, fine-tune, and direct cells complete for three seeds;
- per-session and overall FrameAcc tables are generated from raw counts;
- documentation and the Chinese refactor skill reflect the final behavior.
