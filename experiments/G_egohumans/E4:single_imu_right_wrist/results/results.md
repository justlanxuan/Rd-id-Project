# E4 Results: 4-IMU vs 1-IMU (Right Wrist)

## Overall FrameAcc Comparison

| Setting | 4-IMU | 1-IMU (R_LowArm) | Δ (pp) |
|---|---|---|---|
| Full test (20 seq) | 0.9558 | 0.7934 | 16.25 |
| Train-only subset 1-window (16 seq) | 0.9562 | 0.8080 | 14.81 |
| Train-only subset 4-window vote (16 seq) | 0.9632 | 0.8653 | 9.80 |

## Per-sequence FrameAcc (16 MoBInd-train-only sequences)

| Sequence | 4-IMU 1w | 4-IMU 4w vote | 1-IMU 1w | 1-IMU 4w vote |
|---|---|---|---|---|
| custom_01_011 | 0.9184 | 0.9329 | 0.4501 | 0.4084 |
| custom_02_001 | 0.9360 | 0.9539 | 0.7225 | 0.7422 |
| custom_03_009 | 0.9922 | 0.9922 | 0.7724 | 0.8223 |
| custom_04_011 | 1.0000 | 1.0000 | 0.8528 | 0.9710 |
| custom_05_007 | 0.9072 | 0.9648 | 0.7301 | 0.8581 |
| custom_06_006 | 0.9226 | 0.9260 | 0.6701 | 0.6926 |
| custom_06_019 | 0.9846 | 0.9846 | 0.7889 | 0.9176 |
| custom_06_024 | 0.9326 | 0.9326 | 0.8054 | 0.8528 |
| custom_06_025 | 0.9633 | 0.9708 | 0.8565 | 0.9291 |
| custom_06_036 | 0.9305 | 0.9443 | 0.7945 | 0.8727 |
| custom_06_040 | 0.9969 | 0.9969 | 0.9477 | 0.9969 |
| custom_06_041 | 0.9677 | 0.9677 | 0.9602 | 0.9677 |
| custom_06_054 | 0.9637 | 0.9637 | 0.8246 | 0.9315 |
| custom_06_060 | 0.8854 | 0.8840 | 0.8356 | 0.8840 |
| custom_07_007 | 0.9987 | 0.9987 | 0.9884 | 0.9987 |
| custom_07_011 | 0.9987 | 0.9987 | 0.9285 | 0.9987 |

## AI Reflection

- The 1-IMU model is a strict ablation: only the right-wrist sensor is kept; all other hyperparameters and data splits are identical.
- We expect a measurable but hopefully small drop in FrameAcc, because wrist motion is usually highly correlated with full-body motion in EgoHumans activities.
- Human review: Does this drop justify the hardware simplification, or should we explore fusing wrist + one other sensor as a middle ground?
