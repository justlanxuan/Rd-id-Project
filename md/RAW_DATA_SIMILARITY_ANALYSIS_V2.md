# Raw Data Similarity Analysis V2 — Train Set Pairs

> **Date**: 2026-06-10  
> **Method**: Compute similarity on TRAIN SET pairs (not test set).  
> **TC**: Train subjects = S1, S2, S3. Pairs = (S1,S2), (S1,S3), (S2,S3).  
> **Custom**: Train sessions = 171423, 171724, 172522. Pair = (P0, P1).  
> All metrics computed on RAW DATA (not embeddings). Skeleton normalized (hip-centered, torso=1, 2D).

---

## Angle 1: Inter-Person Similarity

### How It Was Computed

For each pair of subjects/persons:
1. Load ALL frames from the train set for each subject
2. Randomly sample 1000 frame pairs (one from each subject)
3. Compute per-pair MSE and Cosine Similarity on flattened raw vectors
4. Average across all pairs

### Skeleton Similarity (Normalized 2D)

#### TC Train Set Pairs (3 pairs)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| S1 vs S2 | 0.2998 | **0.7889** |
| S1 vs S3 | 0.5297 | **0.7352** |
| S2 vs S3 | 0.5263 | **0.7123** |
| **Average** | **0.4519** | **0.7455** |

#### Custom Train Set Pair (1 pair)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| P0 vs P1 | 0.1871 | **0.7628** |

#### Comparison

| Metric | TC (avg 3 pairs) | Custom (P0 vs P1) | Difference |
|--------|-----------------|-------------------|------------|
| MSE (mean) | 0.4519 | 0.1871 | Custom **59% lower** |
| MSE (std) | 3.8694 | 1.2480 | — |
| Cosine Similarity (mean) | **0.7455** | **0.7628** | Custom **+2.3%** |
| Cosine Similarity (std) | 0.2955 | 0.3451 | — |

**Conclusion**: On the train set, Custom's two people have skeletons that are **slightly more similar** than TC's subject pairs (CosSim 0.763 vs 0.746), but the difference is **small** (+2.3%). This is much smaller than the +12.5% gap observed on the test set (S4 vs S5), likely because TC's train set includes diverse action types that increase within-subject variance.

**Key insight**: The small gap (+2.3%) alone cannot explain Custom's near-random performance (G2=0.50) vs TC's strong performance (G2=0.87). Skeleton similarity is **not the primary bottleneck**.

---

### IMU Similarity

#### TC Train Set Pairs (3 pairs)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| S1 vs S2 | 30.07 | **-0.3172** |
| S1 vs S3 | 9.74 | **0.6550** |
| S2 vs S3 | 30.70 | **-0.3222** |
| **Average** | **23.51** | **0.0052** |

#### Custom Train Set Pair (1 pair)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| P0 vs P1 | 24.75 | **-0.1258** |

#### Comparison

| Metric | TC (avg 3 pairs) | Custom (P0 vs P1) | Difference |
|--------|-----------------|-------------------|------------|
| MSE (mean) | 23.51 | 24.75 | Similar |
| Cosine Similarity (mean) | **0.0052** | **-0.1258** | Both near **zero** |

**Conclusion**: IMU cross-person similarity is **near zero for both datasets**. The IMU signal does not provide meaningful discrimination between people in either TC or Custom. This is consistent with the corruption ablation finding that `imu_pure_noise` drops performance to Custom's level — the model relies primarily on skeleton for identity recognition.

---

## Angle 2: Action Complexity

### Skeleton Action Complexity

| Metric | TC (avg S1+S2+S3) | Custom (avg P0+P1) | Ratio (TC/Custom) |
|--------|------------------|-------------------|------------------|
| Joint position variance | **0.2216** | **0.0792** | **2.80×** |
| Mean joint speed | **0.0337** | **0.0313** | **1.08×** |
| Speed std | **0.3246** | **0.1322** | **2.45×** |
| Static ratio (% near-zero) | 34.0% | **41.7%** | Custom **+7.7pp** |
| Body spread (max-min range) | **18.10** | **4.60** | **3.93×** |

**Key findings**:
- **Joint variance**: Custom is 2.8× lower — joints move much less
- **Body spread**: Custom is 3.9× lower — subjects stay in a much smaller spatial envelope
- **Static ratio**: Custom is 7.7 percentage points higher — more time spent stationary
- **Speed**: The gap is small (1.08×) because train set includes TC's slow actions (ROM, walking)

### IMU Action Complexity

| Metric | TC (avg S1+S2+S3) | Custom (avg P0+P1) | Ratio (TC/Custom) |
|--------|------------------|-------------------|------------------|
| Channel variance | 5.40 | **8.01** | 0.67× (Custom **higher**) |
| Mean channel speed | **0.2500** | **0.1431** | **1.75×** |
| Speed std | **1.6002** | **0.5062** | **3.16×** |
| Static ratio | 56.0% | 57.1% | Similar |

**Key findings**:
- **IMU speed**: Custom is 1.75× lower — less actual motion
- **IMU speed variability**: Custom is 3.2× lower — movement is more uniform
- **Channel variance**: Custom is **higher** (8.0 vs 5.4) — more noise/sensor drift

---

## Per-Subject / Per-Person Breakdown

### TC Train Subjects

| Subject | Skeleton Speed | Body Spread | IMU Speed |
|---------|---------------|------------|-----------|
| S1 | 0.0286 | 18.43 | 0.257 |
| S2 | 0.0348 | 22.71 | 0.259 |
| S3 | 0.0376 | 13.18 | 0.234 |

### Custom Train Persons

| Person | Skeleton Speed | Body Spread | IMU Speed |
|--------|---------------|------------|-----------|
| P0 | 0.0319 | 3.49 | 0.152 |
| P1 | 0.0307 | 5.71 | 0.135 |

---

## Synthesis: Corrected Conclusions

### What Changed from V1 (Test Set) to V2 (Train Set)

| Claim | V1 (Test Set) | V2 (Train Set) | Verdict |
|-------|--------------|----------------|---------|
| Skeleton similarity gap | +12.5% (0.745 vs 0.838) | +2.3% (0.746 vs 0.763) | **Much smaller** |
| IMU similarity | Both negative | Both near zero | **Consistent** |
| Speed gap | 1.4× | 1.08× | **Smaller** |
| Spread gap | 3.5× | 3.9× | **Consistent** |
| Joint variance gap | 4.2× | 2.8× | **Still large** |

### Updated Root Cause Analysis

**1. Skeleton similarity is NOT the primary bottleneck** ❌

The +2.3% similarity gap (0.763 vs 0.746) is too small to explain the 0.50 vs 0.87 performance gap. If skeleton similarity were the main issue, TC's S2 vs S3 (CosSim=0.712) should also struggle — but the model achieves G2=0.87 on TC.

**2. Action simplicity IS a significant factor** ✅

- Body spread is 3.9× smaller in Custom (4.6 vs 18.1)
- Joint variance is 2.8× smaller (0.079 vs 0.222)
- Static ratio is 7.7pp higher

These metrics indicate that Custom subjects move less, explore less spatial range, and have lower motion variability. **The raw data simply contains less motion information.**

**3. IMU is non-discriminative in BOTH datasets** ⚠️

Cross-person IMU CosSim ≈ 0 for TC and ≈ -0.13 for Custom. The IMU signal does not carry identity information in either dataset. The model succeeds on TC because skeleton provides strong identity cues; it fails on Custom because skeleton cues are weakened by low motion complexity.

**4. The true bottleneck: Low motion complexity weakens skeleton identity signal** 🔴

Even though Custom's skeletons are not dramatically more similar than TC's (CosSim 0.763 vs 0.746), the **reduced motion complexity** means:
- Fewer pose variations to reveal body-proportion differences
- Smaller spatial envelope reduces kinematic footprint
- Higher static ratio means more frames with zero discriminative content

The model needs **dynamic motion** to infer identity from skeleton. When subjects stand still or move minimally, even people with different body proportions look identical in pose space.

---

## Bottom Line (Corrected)

> **Custom data fails primarily because the subjects' movements are too simple and static**, not because their skeletons are too similar. The 3.9× smaller body spread, 2.8× lower joint variance, and 7.7pp higher static ratio mean the raw recordings contain **insufficient motion dynamics** for the model to extract identity-discriminating features. Skeleton similarity is a minor secondary factor (+2.3%), and IMU is non-discriminative in both datasets.
>
> **Recommendation**: To improve performance, redesign data collection to:
> 1. **Require continuous, dynamic movement** (minimize static periods)
> 2. **Encourage large spatial exploration** (walking, arm gestures, body turns)
> 3. **Use subjects with visibly different body types** (secondary factor, but helpful)
> 4. **Fix or ignore IMU** — it does not provide identity signal in either dataset
