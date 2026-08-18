# E7 Test Contract

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E7:gap_manifest/scripts/E1_build_final_gap_manifest.py
```

Expected artifact: `/data/fzliang/reid-project/g9/g9_final_gap_manifest.json`.

The manifest must fail if any required E1/E2/E3 evidence file is missing, including `s06_sweep_summary.json`, `custom_imu_filter_control.json`, `s06_prediction_stratification.json`, and `g6_representation_boundary.json`; record all input hashes, preserve `included/conditional/pending`, include the 12-cell S06 coordinate control plus four-session IMU and S06 prediction strata, and preserve the explicit xy-only/full-xyz and S06-independent-ID protocol boundaries.
