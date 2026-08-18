# E2/E4/E5 Test Contract

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E2:multimodal_motion_diagnostics/scripts/B1_build_multimodal_diagnostics.py
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E2:multimodal_motion_diagnostics/scripts/B2_tracking_quality.py
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E2:multimodal_motion_diagnostics/scripts/C1_compare_imu_7d_contract.py
```

Expected artifacts:

- `/data/fzliang/reid-project/g9/e2_multimodal/multimodal_motion_diagnostics.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/tracking_quality.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/imu_contract_comparison.json`

The scripts are read-only with respect to source data and must report zero read/shape/frame mismatches for the included subset. Any missing visibility or independent track ID is reported as a limitation rather than imputed.

Custom detector-ID audit:

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G9:source_to_custom_gap/E2:multimodal_motion_diagnostics/scripts/D8_audit_custom_detector_ids.py
```

Expected artifact: `/data/fzliang/reid-project/g9/e2_multimodal/custom_detector_id_audit.json`, with four sessions, zero duplicate detector-ID/frame rows, and an explicit S06 no-independent-ID limitation.
