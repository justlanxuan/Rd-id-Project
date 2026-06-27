# Raw Data Similarity Analysis: TC vs Custom

> **Date**: 2026-06-10  
> **Goal**: Determine whether Custom's poor performance is due to (1) high inter-person similarity or (2) action complexity issues.

---

## Methodology

- **TC data**: `scale_170` test set (S4, S5, 5 sessions each, ~3.4 min total)
- **Custom data**: 4 sessions, 2 persons each, ~3.4 min total
- **Skeleton**: Extracted skeletons, normalized to hip-centered + torso-length=1, using only (x,y) coordinates
- **IMU**: Raw 48D IMU vectors (4 sensors × 12 channels)
- **No embeddings used** — all metrics computed on raw/preprocessed data

---

## Angle 1: Inter-Person Similarity (Are the two people too similar?)

### Skeleton Similarity (Normalized 2D)

| Metric | TC (S4 vs S5) | Custom (P0 vs P1) | Difference |
|--------|--------------|-------------------|------------|
| MSE (mean) | **0.4241** | **0.1094** | Custom **74% lower** |
| MSE (std) | 1.5421 | 0.5049 | — |
| Cosine Similarity (mean) | **0.7446** | **0.8375** | Custom **+12.5%** |
| Cosine Similarity (std) | 0.3232 | 0.3162 | — |

**Interpretation**: Custom's two people have **significantly more similar skeletons** than TC's S4 vs S5. The cosine similarity of 0.84 means their normalized skeletons are highly correlated — their body proportions and pose patterns are very alike.

### IMU Similarity

| Metric | TC (S4 vs S5) | Custom (P0 vs P1) | Difference |
|--------|--------------|-------------------|------------|
| MSE (mean) | **23.50** | **38.18** | Custom **+62% higher** |
| MSE (std) | 35.05 | 39.23 | — |
| Cosine Similarity (mean) | **0.3179** | **-0.5250** | Custom strongly **negative** |
| Cosine Similarity (std) | 0.2501 | 0.4388 | — |

**Interpretation**: Custom's IMU data shows **negative cosine similarity** (-0.53), meaning the two people's IMU vectors point in roughly opposite directions. This suggests the IMU data contains **strong session-specific or person-specific bias** (e.g., sensor mounting orientation, calibration differences) rather than genuine motion signatures. The high MSE (38 vs 23) further confirms the IMU data is noisier and less consistent in Custom.

### Within-Person vs Cross-Person Variance

To understand whether cross-person differences are distinguishable from within-person (session-to-session) differences:

| Dataset | Modality | Within-Person MSE | Cross-Person MSE | Ratio (cross/within) |
|---------|----------|------------------|------------------|---------------------|
| TC | Skeleton | 1.08 (S4 session-to-session) | 0.65 (S4 vs S5) | **0.61×** |
| TC | IMU | 26.07 (S4 session-to-session) | 27.16 (S4 vs S5) | **1.04×** |
| Custom | Skeleton | 0.13 (P0 session-to-session) | 0.13 (P0 vs P1) | **1.04×** |
| Custom | IMU | 26.67 (P0 session-to-session) | 30.03 (P0 vs P1) | **1.13×** |

**Key insight for TC**: The within-subject skeleton MSE (1.08) is **higher** than cross-subject MSE (0.65). This is because TC's sessions are *different action types* (acting, freestyle, ROM, walking) — the same person looks very different across sessions. Yet the model achieves G2=0.87 because it learns **identity-invariant features** (body proportions, gait style) that persist across actions.

**Key insight for Custom**: The cross-person skeleton MSE (0.13) is virtually identical to within-person MSE (0.13). **The two people do not look any more different from each other than one person looks across different sessions.** This is the core problem: Custom's subjects lack distinguishing body characteristics.

---

## Angle 2: Action Complexity (Too simple or too complex?)

### Skeleton Action Complexity

| Metric | TC (avg S4+S5) | Custom (avg P0+P1) | Ratio (TC/Custom) |
|--------|---------------|-------------------|------------------|
| Joint position variance | **0.2836** | **0.0673** | **4.2×** |
| Mean joint speed | **0.0407** | **0.0289** | **1.4×** |
| Speed std | **0.3017** | **0.1172** | **2.6×** |
| Static ratio (% frames with near-zero motion) | **31.0%** | **39.2%** | — |
| Body spread (max-min position range) | **16.16** | **4.62** | **3.5×** |

**Interpretation**: Custom movements are **dramatically simpler** than TC:
- Joint variance is **4.2× lower** — joints move much less
- Body spread is **3.5× lower** — subjects stay in a smaller spatial envelope
- Static ratio is **8.2% higher** — more time spent not moving
- Speed variability is **2.6× lower** — movement is more uniform and less dynamic

### IMU Action Complexity

| Metric | TC (avg S4+S5) | Custom (avg P0+P1) | Ratio (TC/Custom) |
|--------|---------------|-------------------|------------------|
| Channel variance | **8.84** | **10.60** | 0.83× |
| Mean channel speed | **0.388** | **0.226** | **1.7×** |
| Speed std | **2.577** | **0.869** | **3.0×** |
| Static ratio | **59.8%** | **57.5%** | — |

**Interpretation**: Custom IMU shows **higher variance but lower speed**. This suggests:
- Higher variance = more noise or sensor drift
- Lower speed = less actual motion
- The IMU data contains **noise without signal** — high variance from sensor artifacts, not from meaningful motion

### Per-Session Breakdown

#### TC (S4) — Diverse Action Types

| Session | Skeleton Speed | Static Ratio | Body Spread | IMU Speed |
|---------|---------------|-------------|------------|-----------|
| acting3 | 0.028 | 21.7% | 1.66 | 0.674 |
| freestyle1 | **0.131** | 47.9% | **26.73** | **1.005** |
| freestyle3 | 0.128 | 39.2% | 11.61 | 0.478 |
| rom3 | 0.019 | 40.0% | 3.01 | 0.085 |
| walking2 | 0.030 | 19.6% | 1.11 | 0.406 |

TC sessions span a **huge range** of complexity: freestyle has 4–7× higher speed and 7–24× larger spread than ROM/walking.

#### Custom — Uniformly Low Complexity

| Session | Person | Skeleton Speed | Static Ratio | Body Spread | IMU Speed |
|---------|--------|---------------|-------------|------------|-----------|
| 171423 | 0 | 0.048 | 23.4% | 1.17 | 0.219 |
| 171423 | 1 | 0.035 | 42.7% | 1.37 | 0.179 |
| **171724** | 0 | **0.006** | **66.5%** | **0.22** | **0.018** |
| **171724** | 1 | **0.009** | **57.5%** | **0.26** | **0.032** |
| 172257 | 0 | 0.021 | 27.1% | 0.45 | 0.560 |
| 172257 | 1 | 0.019 | 32.3% | 0.31 | 0.463 |
| 172522 | 0 | 0.041 | 27.5% | 3.28 | 0.212 |
| 172522 | 1 | 0.047 | 26.9% | 5.49 | 0.187 |

**Session 171724 is catastrophically static**: 57–67% of frames show near-zero motion, body spread is 5× smaller than other sessions, and IMU speed is 10× lower. This session alone could collapse model performance.

---

## Synthesis: Why Custom Fails

### Primary Cause: Insufficient Inter-Person Discriminability

1. **Skeletons are too similar**: Custom's two people have cosine similarity of 0.84 (vs TC's 0.74). Even after normalization, their body proportions and pose patterns are highly correlated.

2. **Cross-person variance ≈ within-person variance**: In Custom, the difference between Person 0 and Person 1 (MSE=0.13) is no larger than Person 0's own variation across sessions (MSE=0.13). **The model has no signal to learn from.**

### Secondary Cause: Action Complexity is Too Low

3. **Movements are 3–4× simpler**: Lower joint variance, smaller body spread, higher static ratio, and less speed variability all indicate that Custom subjects perform minimal, repetitive motions.

4. **Session 171724 is nearly static**: 2/3 of frames show no meaningful movement. This session contributes ~25% of training data but carries ~0% identity information.

### Tertiary Cause: IMU Data is Noisy and Inconsistent

5. **Negative IMU cosine similarity** (-0.53) suggests sensor bias dominates over motion signal.
6. **High IMU variance with low speed** indicates noise without meaningful motion content.

---

## Bottom Line

> **Custom data fails because the two people are too similar in both body structure and movement pattern, while their movements are simultaneously too simple to provide discriminating motion signatures.**
>
> The TC dataset succeeds not because it has more data, but because:
> 1. S4 and S5 have genuinely different body proportions and gait styles
> 2. They perform diverse, dynamic actions that reveal individual movement characteristics
> 3. Even when cross-subject raw similarity is high, the model can extract latent identity features from motion dynamics
>
> **Recommendation**: To improve Custom performance, redesign the data collection to:
> 1. **Use subjects with visibly different body types** (height, limb proportions)
> 2. **Instruct subjects to perform distinct, dynamic actions** (e.g., walking at different speeds, different arm gestures, different movement styles)
> 3. **Eliminate or minimize static periods** — ensure continuous motion throughout recording
> 4. **Verify IMU sensor calibration** — negative cosine similarity suggests mounting/calibration issues
