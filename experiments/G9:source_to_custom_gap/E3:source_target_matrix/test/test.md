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

Custom target IMU filter control:

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D5_run_custom_imu_filter_control.py \
  --device cuda:0
```

Expected artifact: `/data/fzliang/reid-project/g9/e3_source_target/custom_imu_filter_control.json`, containing raw, invalid-fill-only, and unit-normalized results for four sessions with identical denominators.

S06 prediction strata:

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D6_stratify_s06_predictions.py
```

Expected artifact: `/data/fzliang/reid-project/g9/e3_source_target/s06_prediction_stratification.json`, with six methods, two variants, 528 sequences and no missing joins.

Representation-boundary audit:

```bash
PYTHONPATH=. /home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E3:source_target_matrix/scripts/D7_audit_g6_representation_boundary.py
```

Expected artifact: `/data/fzliang/reid-project/g9/e3_source_target/g6_representation_boundary.json`, with equal xy/different z inputs and zero max difference in both G6 skeleton feature paths.

The index must contain the existing G6 protocol hash, run-record provenance, raw `correct/total`, target-session feature joins, an explicit availability/gate status for every candidate skeleton source, and a list of cells not yet benchmarked. D4, D5 and D6 must preserve raw counts, controlled deltas, denominators, strata definitions and representation limitations.
