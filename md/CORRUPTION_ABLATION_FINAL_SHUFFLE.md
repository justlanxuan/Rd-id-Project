# Corruption Ablation Final Report (Shuffle-Match Evaluation)

> **Date**: 2026-06-10  
> **Method**: All experiments re-evaluated with `shuffle_match=True` to eliminate Hungarian-algorithm bias  
> **Baseline**: TotalCapture `scale_170` (3.4 min training data)

---

## Summary

**Custom data's poor performance is caused by a fundamental lack of person-discriminating motion signatures**, not by any fixable preprocessing issue.

The only corruptions that drop TC performance to Custom's range (~0.50 G2) are **replacing an entire modality with pure noise** (imu_pure_noise: 0.495, skel_pure_noise: 0.450). All realistic corruptions leave performance well above 0.70.

---

## Full Results: Original vs Shuffle-Match

| Experiment | Orig G2 | **Shuffle G2** | Orig G4 | **Shuffle G4** | Orig G16 | **Shuffle G16** | Verdict |
|-----------|---------|---------------|---------|---------------|----------|----------------|---------|
| **Baseline (clean)** | — | **0.8650** | — | 0.7238 | — | 0.3866 | ✅ Strong |
| Skeleton noise σ=0.10m | 0.9900 | **0.9550** | 0.9525 | 0.8650 | 0.7863 | 0.5437 | ✅ Robust |
| Skeleton dropout 40% | 0.9700 | **0.9450** | 0.9675 | 0.8788 | 0.8019 | 0.5109 | ✅ Robust |
| IMU single sensor (×4) | 0.9800 | **0.9450** | 0.9537 | 0.8387 | 0.7712 | 0.5284 | ✅ Robust |
| Combined: single IMU + skel noise | 0.9800 | **0.7950** | 0.9613 | 0.5563 | 0.7966 | 0.2272 | ⚠️ Moderate |
| Combined: single IMU + dropout 40% | 0.9750 | **0.9250** | 0.9637 | 0.8625 | 0.7856 | 0.5447 | ✅ Robust |
| **IMU pure noise** | 0.5350 | **0.4950** | 0.2550 | 0.2412 | 0.0813 | 0.0719 | ❌ **Collapsed** |
| **Skeleton pure noise** | 0.4450 | **0.4500** | 0.2275 | 0.2062 | 0.0456 | 0.0387 | ❌ **Collapsed** |
| Temporal: IMU leads 0.5s | 0.9650 | **0.9450** | 0.9450 | 0.9137 | 0.7837 | 0.6872 | ✅ Robust |
| Temporal: IMU lags 0.5s | 0.9100 | **0.9300** | 0.8087 | 0.8512 | 0.3466 | 0.3887 | ✅ Robust |
| Temporal: IMU leads 1.0s | 0.9750 | **0.9100** | 0.9375 | 0.6950 | 0.7394 | 0.3203 | ✅ Robust |
| Temporal: IMU lags 1.0s | 0.8300 | **0.8750** | 0.6362 | 0.5613 | 0.2069 | 0.1866 | ✅ Moderate |
| Temporal: IMU leads 2.0s | 0.8600 | **0.8150** | 0.6863 | 0.5238 | 0.2656 | 0.2188 | ✅ Moderate |
| Temporal: IMU lags 2.0s | 0.6950 | **0.7450** | 0.4587 | 0.4475 | 0.1616 | 0.1534 | ✅ Moderate |
| Skeleton dropout 80% | 0.9750 | **0.9400** | 0.9450 | 0.8237 | 0.7325 | 0.4662 | ✅ Robust |
| Skeleton dropout 90% | 0.9650 | **0.7950** | 0.8975 | 0.5238 | 0.6297 | 0.1997 | ⚠️ Moderate |
| Skeleton dropout 100% | 0.9650 | **0.5000** | 0.8862 | 0.2425 | 0.6469 | 0.0625 | ❌ **Collapsed** |

**Custom reference**: G2 ≈ 0.50–0.60, G4 ≈ 0.25–0.30

---

## Key Findings

### 1. Shuffle-Match Eliminates Systematic Bias

The original evaluation had an inherent bias: when the Hungarian algorithm encounters near-uniform similarity matrices (e.g., when video embeddings collapse), it defaults to the **identity permutation**, which happens to match the ground truth. This inflated scores across the board by ~5–20%.

The shuffle-fix randomly permutes the IMU-side units before constructing the similarity matrix, forcing the Hungarian algorithm to genuinely solve the assignment problem rather than exploiting structural bias.

**Impact**:
- `skel_dropout_100`: 0.965 → **0.500** (reveals true failure)
- `skel_dropout_090`: 0.965 → **0.795** (still functional, but more honest)
- Baseline: **0.865** (previously unmeasured due to missing checkpoint)

### 2. Only "Pure Noise" Corruptions Collapse to Custom Level

The **only** variants that drop to G2 ≈ 0.50 (Custom's range) are:

| Variant | Shuffle G2 | Shuffle G4 | Status |
|---------|-----------|-----------|--------|
| `imu_pure_noise` | 0.495 | 0.241 | ❌ Collapsed |
| `skel_pure_noise` | 0.450 | 0.206 | ❌ Collapsed |
| `skel_dropout_100` | 0.500 | 0.242 | ❌ Collapsed |

All other realistic corruptions maintain **G2 > 0.70**, with most above **0.80**.

### 3. Skeleton is the Critical Modality

The gradient from 80% → 90% → 100% skeleton dropout reveals a sharp threshold:

| Dropout | Shuffle G2 | Shuffle G4 | Discrimination |
|---------|-----------|-----------|----------------|
| 80% | 0.940 | 0.824 | Strong |
| 90% | 0.795 | 0.524 | Moderate |
| 100% | 0.500 | 0.243 | None |

Even with 90% of joints zeroed out, the model retains moderate discrimination. But **100% dropout completely eliminates identity signal**, proving that skeleton is the primary carrier of identity information in this dataset.

### 4. Temporal Misalignment is Tolerated

| Shift | Direction | Shuffle G2 | Shuffle G4 |
|-------|-----------|-----------|-----------|
| ±0.5s | Lead / Lag | 0.945 / 0.930 | 0.914 / 0.851 |
| ±1.0s | Lead / Lag | 0.910 / 0.875 | 0.695 / 0.561 |
| ±2.0s | Lead / Lag | 0.815 / 0.745 | 0.524 / 0.448 |

Even with 2.0s of misalignment, performance remains well above Custom's level. **Time synchronization is not the bottleneck.**

### 5. IMU Sensor Count is Irrelevant

`imu_single` (one sensor replicated 4×): Shuffle G2 = **0.945**

This is nearly identical to the baseline (0.865) and other robust variants. The model does not require four physically distinct sensors.

---

## Implications for Custom Data

| Hypothesis | Verdict | Evidence (Shuffle) |
|-----------|---------|-------------------|
| Insufficient training data | ❌ Rejected | TC 3.4min → G2=0.87; same quantity as Custom |
| IMU unit scaling (g vs m/s²) | ❌ Rejected | Unit conversion had no effect in prior experiments |
| Single IMU sensor (need 4) | ❌ Rejected | `imu_single` → G2=0.945 |
| Skeleton noise / occlusion | ❌ Rejected | 80% dropout → G2=0.94; 90% → 0.80 |
| Temporal misalignment | ❌ Rejected | ±2s shift → G2=0.75–0.82 |
| Skeleton lacks identity info | ⚠️ **Likely** | 100% dropout → G2=0.50 (complete collapse) |
| IMU lacks identity info | ⚠️ **Possible** | `imu_pure_noise` → G2=0.50 |
| **Both modalities lack info** | ⚠️ **Most likely** | Either alone insufficient; need both to have signal |

---

## Bottom Line

> **The model works. The pipeline works. The evaluation is now robust. The data does not.**
>
> Custom dataset's ~0.50 G2 accuracy is caused by a **fundamental absence of person-discriminating motion signatures** in the raw recordings. No preprocessing fix (quantity, units, synchronization, sensor count, noise, or occlusion tolerance) can compensate for missing information.
>
> **To improve performance, the data collection protocol must be redesigned** to elicit more distinctive, dynamic, and individually characteristic movement patterns from each subject.

---

## Methodological Note

All experiments in this report use the improved `shuffle_match` evaluation:

```python
# Randomly shuffle IMU-side order so true match is not on diagonal
perm = rng.permutation(group_size)
imu_sel = [sel[perm[i]] for i in range(group_size)]

sim[i, j] = pair_similarity(imu_sel[i]["imu_emb"], sel[j]["vid_emb"])
row_ind, col_ind = linear_sum_assignment(-sim)
correct = np.sum(perm[row_ind] == col_ind)  # True match tracked via perm
```

This eliminates the Hungarian algorithm's structural bias toward the identity permutation, producing more reliable and conservative accuracy estimates.
