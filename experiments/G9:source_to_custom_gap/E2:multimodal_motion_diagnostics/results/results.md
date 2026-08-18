# E2/E4/E5 Results：可信子集诊断

状态：`screening_plus_local_controls_complete`；D5/D6/D8 已补充固定检查点、prediction strata 和 Custom detector-ID 对照。结果仍不外推为重训后的全域因果结论。

产物：

- `/data/fzliang/reid-project/g9/e2_multimodal/multimodal_motion_diagnostics.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/tracking_quality.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/imu_contract_comparison.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/custom_detector_id_audit.json`

## Coverage and representation

| Source | Records | Frames | Representation | IMU layout | Read/frame errors |
|---|---:|---:|---|---|---:|
| TotalCapture GT | 46 | 179320 | 3D xyz | 7D acc+quat | 0 |
| EgoHumans canonical | 30 | 13658 | 2D xy+visibility | 7D acc+quat | 0 |
| Custom canonical | 7380 | 177120 | 2D xy | 7D acc+quat | 0 |
| S06 AlphaPose | 88 | 46847 | 2D xy+zero-z/visibility | legacy48→L_LowArm acc | 0 |
| S06 FMPose3D | 88 | 46847 | 3D xyz | legacy48→L_LowArm acc | 0 |
| S06 MotionAGFormer | 88 | 46847 | 3D xyz | legacy48→L_LowArm acc | 0 |
| S06 TCPFormer | 88 | 46847 | 3D xyz | legacy48→L_LowArm acc | 0 |
| S06 WHAM | 88 | 46847 | 3D xyz | legacy48→L_LowArm acc | 0 |

## IMU distribution finding

The trusted canonical 7D streams have similar median acceleration energy (approximately 125–129 in the current units), while S06 legacy baseline extraction has median selected-sensor acceleration energy about 22.8. This is a real representation/unit-path difference to control before attributing results to skeleton algorithms; it is not evidence that S06 sensors are physically less active. Custom raw CSV acceleration is converted from g to SI-like units in preprocessing, while S06 uses a legacy matrix/acceleration layout.

After explicit legacy48→7D conversion, quaternion norms are near unit for TotalCapture, EgoHumans and S06. Custom has a small but real invalid-quaternion tail: 1,612 / 177,120 frames (0.91%) have norm outside `[0.9,1.1]`, including 28 zero-norm frames, spread over 892 windows. Therefore Custom remains included for skeleton-target analysis, but fusion/IMU analysis must filter or explicitly model these frames and report the filtered denominator.

## Motion complexity screening

All sources have low/mid/high per-source motion-energy tertiles in the machine-readable artifact. Median bone-normalized motion energy is approximately:

| Source | Low | Mid | High |
|---|---:|---:|---:|
| TotalCapture GT | 0.0087 | 0.0225 | 0.0286 |
| EgoHumans 2D | 0.0835 | 0.1648 | 0.2762 |
| Custom 2D | 0.0231 | 0.0478 | 0.0887 |
| AlphaPose 2D | 0.1050 | 0.1519 | 0.2336 |
| FMPose3D | 0.1652 | 0.2080 | 0.2902 |
| MotionAGFormer | 0.0463 | 0.0623 | 0.1271 |
| TCPFormer | 0.0769 | 0.1071 | 0.2047 |
| WHAM | 0.0744 | 0.1083 | 0.1700 |

These are within-source tertiles and must not be read as a cross-space ranking. They show that a complexity-matched evaluation is necessary: Custom windows have lower normalized motion energy than most S06 lifted outputs, while EgoHumans 2D and AlphaPose contain larger screen-space motion after bone normalization.

## Cross-modal lag screening

The median best lag / absolute correlation is: TotalCapture `+1 / 0.395`, EgoHumans `-1 / 0.137`, Custom `0 / 0.537`, AlphaPose `0 / 0.277`, FMPose3D `+1 / 0.269`, MotionAGFormer `-1 / 0.307`, TCPFormer `+1 / 0.286`, WHAM `0 / 0.326`. The lag spread is several frames for EgoHumans and S06; this supports an explicit lag/session analysis, but it does not establish causality.

## Tracking and identity

- S06 visibility coverage means: AlphaPose 0.944, FMPose3D/MotionAGFormer/TCPFormer 0.981, WHAM 0.921. These are computed from output visibility, not inferred from finite coordinates.
- S06 output identity is `inherited_gt_person_order`; independent ID switches cannot be identified because no independent track IDs are stored.
- Custom has 7380 GT window rows, all `skeleton_source=gt`, with zero person/IMU mapping mismatches. D8 audits the raw Custom AlphaPose JSON: four sessions have no duplicate ID/frame rows; raw-ID→GT transition counts are 12 and GT→raw-ID transition counts are 53 under per-frame Hungarian IoU≥0.1. These are transition diagnostics, not relinking-based definitive ID-switch labels. S06 remains unobservable because its NPZ outputs contain inherited person order and no independent IDs.

## Fixed-checkpoint controls

- D5 raw versus invalid-quaternion-fill-only versus unit-normalized Custom target IMU: aggregate history FrameAcc 0.54714 → 0.53671 (−160/15349 correct), with the change concentrated in `20260211_172257` (15.26% invalid quaternion frames); instantaneous FrameAcc is unchanged.
- D6 connects all 528 S06 predictions to pooled motion-energy, visibility-coverage and fragmentation-proxy strata. It reports correct/total per source/variant/bucket and keeps sparse buckets explicit.
- D8 connects Custom AlphaPose raw detector IDs to GT boxes without altering tracks; S06 ID-switch attribution remains explicitly unavailable.

## Remaining protocol boundary

The current G6 encoder consumes xy only (D7 max feature difference under arbitrary z change is 0). A full-xyz source attribution therefore requires a new xyz-compatible encoder/protocol; it is not silently inferred from the existing xy projection.
