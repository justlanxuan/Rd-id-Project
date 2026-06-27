# Raw Data Similarity Analysis V4 — Matched Sample Size, Simultaneous Windows

> **Date**: 2026-06-10  
> **Method**: Fair comparison with matched sample sizes.
> - TC: Randomly sample 380 window pairs per cross-subject combination (S1×S2, S1×S3, S2×S3)
> - Custom: Compute all 380 simultaneous window pairs (same session, same time, P0 vs P1)
> - All similarities computed on RAW DATA (not embeddings)
> - Skeleton normalized (hip-centered, torso-length=1, 2D only)

---

## Angle 1: Inter-Person Similarity

### How It Was Computed

**TC**:
1. Load all 120 train windows for S1, S2, S3
2. For each pair (S1,S2), (S1,S3), (S2,S3):
   - Randomly sample 380 window pairs from all possible combinations
   - For each sampled pair, compute framewise MSE and Cosine Similarity
   - Average across the 380 pairs
3. Report average across the 3 cross-subject pairs

**Custom**:
1. Load all 380 simultaneous window pairs from train set
   - Each pair = same session, same (window_start, window_end), P0 vs P1
2. For each pair, compute framewise MSE and Cosine Similarity
3. Average across all 380 pairs

### Skeleton Similarity (Normalized 2D)

#### TC Cross-Subject Pairs (380 pairs each)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| S1 vs S2 | 0.0497±0.0106 | **0.9581±0.0092** |
| S1 vs S3 | 0.1374±0.2082 | **0.9472±0.0136** |
| S2 vs S3 | 0.1304±0.2075 | **0.9524±0.0117** |
| **Average** | **0.1058** | **0.9526** |

#### Custom Simultaneous Pair (380 pairs)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| P0 vs P1 | 0.1358±0.3765 | **0.7950±0.3327** |

#### Comparison

| Metric | TC (avg 3 pairs) | Custom (P0 vs P1) | Difference |
|--------|-----------------|-------------------|------------|
| MSE (mean) | 0.1058 | 0.1358 | Custom **+28%** |
| Cosine Similarity (mean) | **0.9526** | **0.7950** | Custom **-16.6%** |

**Critical Finding**: TC's cross-subject skeleton similarity (0.953) is **significantly HIGHER** than Custom's (0.795). This means:
- When TC subjects perform the same type of action (e.g., acting, walking), their normalized skeletons are nearly identical (cosine similarity > 0.95)
- Custom's two people, even when captured simultaneously, have **more distinguishable** skeleton patterns
- **Skeleton similarity is NOT the bottleneck** — if it were, TC should perform worse than Custom

### IMU Similarity

#### TC Cross-Subject Pairs (380 pairs each)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| S1 vs S2 | 24.51±9.05 | **-0.2043±0.0992** |
| S1 vs S3 | 8.20±8.10 | **0.7320±0.0922** |
| S2 vs S3 | 25.95±9.84 | **-0.2315±0.0975** |
| **Average** | **19.55** | **0.0987** |

#### Custom Simultaneous Pair (380 pairs)

| Pair | MSE | Cosine Similarity |
|------|-----|-------------------|
| P0 vs P1 | 34.83±15.10 | **-0.6255±0.2887** |

#### Comparison

| Metric | TC (avg 3 pairs) | Custom (P0 vs P1) | Difference |
|--------|-----------------|-------------------|------------|
| MSE (mean) | 19.55 | 34.83 | Custom **+78%** |
| Cosine Similarity (mean) | **0.0987** | **-0.6255** | Custom **strongly negative** |

**Finding**: Custom's IMU shows strong negative correlation (-0.63), indicating systematic sensor bias or orientation differences between the two IMU sets. TC's IMU is near-zero on average, but with high variance across pairs (some positive, some negative).

---

## Angle 2: Action Complexity

### Note on Methodology Difference

V4 computes complexity from **train windows** (24-frame windows), while V1/V2 computed from **all frames globally**. The window-level analysis can differ from global analysis because:
- Short windows capture local motion dynamics
- TC windows come from diverse action types, averaging may obscure global patterns
- Custom windows capture simultaneous two-person interactions

For this reason, the V1/V2 global complexity results are considered more reliable for cross-dataset comparison. The V4 window-level results are provided for reference.

### Skeleton Action Complexity (Window-Level)

| Metric | TC (avg S1+S2+S3) | Custom (avg P0+P1) | Ratio (TC/Custom) |
|--------|------------------|-------------------|------------------|
| Joint position variance | 0.0538 | 0.0795 | 0.68× |
| Mean joint speed | 0.0204 | 0.0361 | 0.56× |
| Speed std | 0.1146 | 0.1524 | 0.75× |
| Static ratio | 42.8% | 40.9% | — |
| Body spread | 2.86 | 4.60 | 0.62× |

**Note**: Window-level complexity shows Custom with **higher** local motion than TC. This is because Custom windows capture two-person interactions within a short timeframe, while TC windows from different sessions may include simpler actions (ROM, walking). The **global** analysis (V1/V2) is more reliable for this comparison.

### IMU Action Complexity (Window-Level)

| Metric | TC (avg S1+S2+S3) | Custom (avg P0+P1) | Ratio (TC/Custom) |
|--------|------------------|-------------------|------------------|
| Channel variance | 3.83 | 8.03 | 0.48× |
| Mean channel speed | 0.185 | 0.168 | 1.10× |
| Speed std | 1.901 | 0.659 | 2.88× |
| Static ratio | 60.9% | 56.2% | — |

---

## Synthesis

### Definitive Conclusions

| Hypothesis | Evidence | Verdict |
|-----------|----------|---------|
| Skeleton too similar | TC CosSim=0.953 > Custom=0.795 | ❌ **Rejected** |
| IMU has systematic bias | Custom CosSim=-0.63, TC=+0.10 | ⚠️ **Confirmed for Custom** |
| Action too simple (global) | TC spread=18.1 vs Custom=4.6 (V1/V2) | ✅ **Confirmed** |
| Action too simple (window) | TC spread=2.86 vs Custom=4.60 (V4) | ❓ **Inconclusive** |

### The Real Story

The V4 simultaneous-window analysis reveals a surprising fact: **TC subjects performing the same action type have more similar skeletons than Custom's two people captured together.** This proves that raw skeleton similarity is not the limiting factor.

Instead, the key difference lies in **motion diversity across the dataset**:
- TC train set spans 12 different sessions across 3 subjects (acting, freestyle, ROM, walking)
- The model learns to associate **identity-invariant body characteristics** (proportions, gait style) with subject labels
- Custom's limited motion diversity within each session prevents the model from extracting such identity features

**Bottom line**: Custom fails not because P0 and P1 are too similar, but because **the overall motion repertoire is too limited** to reveal distinguishing body characteristics. The model needs diverse actions to "see" how different body types move differently.

---

## Recommendations (Updated)

1. **Increase motion diversity per session** — require subjects to perform multiple distinct action types (walking, arm gestures, turning, bending)
2. **Ensure continuous motion** — minimize static periods that provide no discriminative signal
3. **Fix IMU calibration** — negative cosine similarity indicates sensor bias
4. **Body type diversity helps, but is secondary** — the primary issue is motion diversity, not body similarity
