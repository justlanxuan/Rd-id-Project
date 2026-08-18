# E3 Plan：Source/Target skeleton sweep

## Current evidence

`D1_build_source_target_matrix.py` indexes the existing G6 66 evaluation runs (22 session/condition cells, three seeds where applicable), preserves raw `correct/total`, protocol hash and run-record paths, and joins Custom session motion/IMU diagnostics.

## Controlled next matrix

1. Keep Custom GT target and fixed verified 7D IMU; compare source skeleton tracks separately by 2D/3D representation.
2. Add target-side Custom AlphaPose only after raw output has a verified person/time join; do not use YOLO-Pose high until its outlier policy is fixed.
3. Repeat IMU-only, skeleton-only and fusion controls with invalid-quaternion filtering recorded in the denominator.
4. Stratify predictions by the B1 low/mid/high complexity bins and B2 visibility/tracklet bins.

## Executed control

D3/D4 completed the fixed-checkpoint S06 sweep for all six methods × two coordinate variants across all 88 available Custom sequences. The control fixes baseline IMU, GT person order, checkpoint, window 24 and stride 16; `screen_calibrated` is a diagnostic root/torso coordinate intervention. Results and limits are recorded in `results/results.md` and `/data/fzliang/reid-project/g9/e3_source_target/s06_eval/s06_sweep_summary.json`. Full 2D/3D representation transfer, filtered/unfiltered IMU fusion, and retrained source-domain comparisons remain open.
