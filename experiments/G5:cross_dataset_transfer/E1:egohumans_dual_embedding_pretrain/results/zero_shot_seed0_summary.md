# E1 Zero-Shot Result: seed0

## Setup
- Source: EgoHumans dual-embedding (Model-L + Model-G), seed 0
- Target: custom w24 test clips
- Similarity normalization: none / zscore

## Result

| Norm | Best α | Mean FrameAcc |
|---|---|---|
| none | 1.0 | **0.2940** |
| zscore | 1.0 | **0.2940** |

## Per-α Detail (sim_norm=none)

| α | Mean FrameAcc |
|---|---|
| 0.0 (global only) | 0.2499 |
| 0.1 | 0.2389 |
| 0.2 | 0.2267 |
| 0.3 | 0.2270 |
| 0.4 | 0.2302 |
| 0.5 | 0.2395 |
| 0.6 | 0.2505 |
| 0.7 | 0.2550 |
| 0.8 | 0.2731 |
| 0.9 | 0.2830 |
| 1.0 (local only) | **0.2940** |

## Key Observations

1. **Pure local (α=1.0) is best**: Adding the EgoHumans-trained global branch monotonically hurts performance.
2. **Global branch transfers very poorly**: α=0.0 yields only 0.25 mean FrameAcc.
3. **This zero-shot result (0.294) is below the historical E9 zero-shot (0.339)** using E8 single-IMU checkpoint.
4. **Per-clip variance is huge**:
   - `171423_seg0/seg1`: ~0.40–0.51
   - `171724_seg0/seg1`: ~0.00–0.05 (complete failure)
   - `172257_seg0`: up to 0.64 with local
   - `172522_seg0/seg1`: ~0.28–0.32
5. **Similarity normalization does not help**: zscore gives the same best mean and same best α.

## Implications

- The EgoHumans → custom domain gap remains large for the dual-embedding architecture.
- The global branch (full pose2d) appears to overfit to EgoHumans' camera/view/activity distribution and does not generalize to custom.
- Fine-tune on custom is necessary to assess whether the pre-trained local branch can be adapted to exceed from-scratch performance.
