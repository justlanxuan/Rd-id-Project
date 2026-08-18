# E3 Test Contract

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D1_build_source_target_matrix.py
```

Expected artifact: `/data/fzliang/reid-project/g9/e3_source_target/source_target_matrix.json`.

Prediction strata command:

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D2_stratify_predictions.py
```

Expected artifact: `/data/fzliang/reid-project/g9/e3_source_target/prediction_stratification.json`.

Fixed-checkpoint S06 sweep (full run; can be split by `--methods`, `--variants`, or `--sessions`):

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D3_run_s06_segment_sweep.py \
  --device cuda:0
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D4_summarize_s06_sweep.py
```

Expected artifact: `/data/fzliang/reid-project/g9/e3_source_target/s06_eval/s06_sweep_summary.json`, with 12 cells, 528 paired sequence deltas, and no missing cells. The result is a fixed-checkpoint coordinate intervention with fixed baseline IMU/person order; it must not be reported as retrained source-domain performance.

The index must contain the existing G6 protocol hash, run-record provenance, raw `correct/total`, target-session feature joins, an explicit availability/gate status for every candidate skeleton source, and a list of cells not yet benchmarked. The D4 summary must preserve raw/screen `correct/total`, per-method deltas, and the xy-projection limitation.
