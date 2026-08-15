# Reproducible Benchmarks

The official three-dataset benchmark is G6. Its goal, formulation, plan,
protocol, progress, data manifests, and final reports live under:

```text
experiments/G6:official_refactor_and_three_dataset_benchmark/
```

## Matrix

| Condition | Training jobs | Evaluations |
|---|---:|---:|
| TotalCapture/EgoHumans source | 6 | 6 |
| Source to Custom zero-shot | 0 | 24 |
| Source to Custom fine-tune | 24 | 24 |
| Custom direct LOSO | 12 | 12 |
| Total | 42 | 66 |

Every training condition uses seeds `0`, `42`, and `123`. Custom uses four
held-out sessions and reports each session plus macro-session and
micro/weighted FrameAcc.

## Workflow

1. Build byte-stable data manifests and inspect split/content statistics.
2. Obtain explicit human confirmation of the protocol document.
3. Create the immutable protocol record and hash.
4. Generate all 108 protocol-bound configs.
5. Generate and run two bounded, non-formal one-epoch smoke configs.
6. Dry-run the dependency graph with an explicit GPU list.
7. Execute/resume the formal graph; never overwrite invalid artifacts.
8. Validate all 66 run records and aggregate results automatically.

Run `python -m tools.g6.<command> --help` for exact arguments:

```bash
python -m tools.g6.build_data_manifests --help
python -m tools.g6.lock_protocol --help
python -m tools.g6.build_configs --help
python -m tools.g6.build_smoke_configs --help
python -m tools.g6.run_jobs --help
python -m tools.g6.aggregate_results --help
```

The runner accepts only a locked protocol record. Completion is based on
validated checkpoint/run-record contents and hashes, not path existence. A
partial or corrupt artifact stops the run for human inspection.
