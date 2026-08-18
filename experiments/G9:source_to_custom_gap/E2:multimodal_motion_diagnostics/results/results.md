# E2/E4/E5 Results：可信子集诊断

状态：`screening_complete`；尚未执行模型干预，因此这些结果用于定位假设，不是性能因果结论。

产物：

- `/data/fzliang/reid-project/g9/e2_multimodal/multimodal_motion_diagnostics.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/tracking_quality.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/imu_contract_comparison.json`

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
- Custom has 7380 GT window rows, all `skeleton_source=gt`, with zero person/IMU mapping mismatches; detector tracklet analysis therefore remains pending on raw detector outputs.

## Next intervention

Before source/target performance ranking, run representation controls: (1) convert all included IMU streams to one explicit 7D contract; (2) split 2D and 3D skeleton tracks; (3) root/torso normalize; (4) rerun lag and complexity-matched summaries; then use IMU-only/skeleton-only/fusion controls to test whether the observed marginal and cross-modal gaps explain transfer loss.
