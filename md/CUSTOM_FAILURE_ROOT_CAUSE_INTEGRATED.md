# Custom Failure: Integrated Root Cause Analysis

> **Date**: 2026-06-11  
> **Perspective**: The model's essence is learning cross-modal alignment between IMU and skeleton embeddings.

---

## 1. Model Essence Recap

The alignment model learns:

$$f_{imu}(IMU) \approx f_{skel}(Skeleton) \quad \text{for the same person}$$

and

$$f_{imu}(IMU_i) \neq f_{skel}(Skeleton_j) \quad \text{for different people } i \neq j$$

Training uses cross-modal contrastive/metric learning. Evaluation uses Hungarian matching on learned embeddings to assign IMU windows to skeleton windows within a group.

**For the model to succeed, both modalities must provide stable, person-specific signals that can be aligned.**

---

## 2. Empirical Findings Summary

### 2.1 Performance Gap (Shuffle-Match Evaluation)

| Dataset | Train Duration | G2 Accuracy |
|---------|---------------|-------------|
| TC scale_170 | 3.4 min | **0.865** |
| Custom complete | ~3.4 min | **~0.50** |

### 2.2 Corruption Ablation on TC

Only corruptions that **destroy an entire modality** drop TC to Custom's level:

| Corruption | Shuffle G2 | Meaning |
|-----------|-----------|---------|
| Skeleton pure noise | 0.450 | Model cannot learn identity from skeleton |
| IMU pure noise | 0.495 | Model cannot learn identity from IMU |
| Skeleton 100% dropout | 0.500 | Skeleton signal completely absent |
| All realistic corruptions (noise, occlusion, time shift) | > 0.70 | Model is robust |

**Conclusion**: Custom's performance is at the level of "one modality is pure noise."

### 2.3 Raw Data Similarity (Window-Level, Matched N=380)

| Metric | TC | Custom | Interpretation |
|--------|-----|--------|----------------|
| Skeleton CosSim (cross-person) | 0.953 | 0.795 | TC people have more similar normalized poses |
| IMU CosSim (cross-person) | 0.099 | -0.626 | Custom IMU has strong opposite-direction bias |
| Bone proportion MSE | 0.004 | 0.072 | Custom bone proportions vary 16× more |
| Bone proportion temporal std | 0.01 | 0.20 | Custom skeleton is 15-40× less stable |

### 2.4 Action Complexity (Global Frame Analysis)

| Metric | TC | Custom |
|--------|-----|--------|
| Skeleton body spread | 18.1 | 4.6 |
| Skeleton joint variance | 0.284 | 0.067 |
| Static ratio (skeleton) | 31% | 39% |
| Session 171724 static ratio | — | **67%** |

---

## 3. Why Custom Fails: The Alignment Perspective

The model fails on Custom because **the alignment problem is ill-posed** — one or both modalities lack the stable, person-specific structure needed for cross-modal learning.

### Cause 1: Skeleton Embeddings Are Unstable (Primary)

**Evidence**:
- Bone proportion temporal std: Custom 0.18-0.21 vs TC 0.005-0.013
- Custom uses AlphaPose 2D skeleton; TC uses accurate 3D GT skeleton
- Same person's "body proportions" change by ±20% across frames in Custom

**Alignment impact**:
- The model tries to learn: `f_skel(Skeleton_personA) ≈ constant identity vector`
- But `Skeleton_personA` at frame t=100 has very different bone proportions than at t=200
- So `f_skel(Skeleton_personA)` is not a stable point in embedding space
- Cross-modal contrastive loss cannot converge to a consistent alignment

**Analogy**: It's like trying to match a blurred, constantly changing photograph (Custom skeleton) with a clear audio recording (IMU). The photograph doesn't provide a stable visual identity.

### Cause 2: IMU Has Systematic Cross-Person Bias (Secondary)

**Evidence**:
- Custom P0 vs P1 IMU CosSim = -0.626 (strong opposite-direction bias)
- 94% of window pairs have negative CosSim
- 37% are very negative ([-1.0, -0.8))

**Alignment impact**:
- The model tries to learn: `f_imu(IMU_personA) ≈ f_skel(Skeleton_personA)`
- But IMU vectors for P0 and P1 point in roughly opposite directions in raw feature space
- The encoder must learn to map opposite-direction vectors to the same person or different persons
- With unstable skeleton (Cause 1), this mapping cannot be learned reliably

**Important nuance**: A stable opposite-direction relationship could theoretically be learned (e.g., P0 always positive, P1 always negative). But combined with skeleton instability, the model cannot determine whether the negative similarity is a person-specific signal or noise.

### Cause 3: Insufficient Motion Diversity (Tertiary)

**Evidence**:
- Custom body spread = 4.6 vs TC = 18.1 (3.9× smaller)
- Custom joint variance = 0.067 vs TC = 0.284 (4.2× smaller)
- Session 171724 is 67% static

**Alignment impact**:
- Static or repetitive motions provide very few informative windows
- With limited motion, both IMU and skeleton embeddings collapse to similar vectors
- The contrastive loss has no "hard negatives" — all windows look alike
- The model cannot learn what distinguishes Person A from Person B

**Combined effect**: Even if skeleton were stable and IMU well-calibrated, the lack of diverse motion would still make identity discrimination difficult.

### Cause 4: Cross-Modal Misalignment Becomes Self-Reinforcing

The three causes interact destructively:

```
Unstable Skeleton  ──┐
                     ├──→  Cannot learn stable cross-modal alignment
IMU Direction Bias  ──┤
                     │
Low Motion Diversity ─┘
         │
         ▼
   Embeddings collapse
         │
         ▼
   G2 ≈ 0.50 (random)
```

1. Skeleton instability → skeleton embeddings vary widely for the same person
2. IMU bias → IMU embeddings have systematic but not person-unique structure
3. Low motion diversity → limited signal to disambiguate identity
4. The model enters a shortcut-learning regime: it may exploit session-specific biases (e.g., IMU calibration) instead of learning person identity

This explains the earlier observation of **identity flips**: models achieve high HOTA (~0.94) but near-random FrameAcc (~0.04) by exploiting session-level IMU bias rather than person-specific motion signatures.

---

## 4. Why TC Succeeds Under the Same Logic

In TC, the alignment problem is well-posed:

1. **Stable skeleton**: GT 3D skeleton provides consistent bone proportions (std≈0.01)
2. **Diverse motion**: 12 sessions across 3 subjects with 4 action types
3. **IMU is usable**: Cross-person IMU has mixed but learnable structure

The model can learn:
- `f_skel(S1)` is a stable point in embedding space
- `f_imu(S1)` aligns with `f_skel(S1)` across diverse actions
- `f_skel(S1)`, `f_skel(S2)`, `f_skel(S3)` are well-separated

Even though raw skeleton CosSim between S1/S2 is 0.953, the model extracts subtle but stable identity features from the 3D dynamics.

---

## 5. Counter-Arguments Considered

### Argument: "Custom people have different body proportions, so skeleton should be discriminative"

**Reality check**: Bone proportion analysis shows Custom P0 vs P1 MSE = 0.072, while TC S1 vs S2 MSE = 0.007. Custom's body proportion difference is actually **16× larger** than TC's.

But this difference is **not stable over time** (temporal std 0.20 vs 0.01). The model cannot learn from a feature that changes by ±20% for the same person across frames.

### Argument: "Custom IMU direction bias could be a useful identity signal"

**Reality check**: P0 vs P1 IMU is consistently negative (94%). This is a strong signal. However:
- The model does not know a priori that P0=positive, P1=negative
- It must learn this from skeleton-IMU alignment
- With unstable skeleton, the alignment signal is too noisy
- The model may overfit to session-specific biases instead of generalizable person identity

### Argument: "Is it just insufficient training data?"

**Reality check**: TC scale_170 (3.4 min) achieves G2=0.865. Custom has similar duration. The corruption ablation shows that quantity is not the issue — **information content** is.

---

## 6. Bottom Line

> **Custom fails because the cross-modal alignment problem is fundamentally broken.**
>
> The skeleton modality (AlphaPose 2D) is too unstable over time to provide a reliable visual identity signal. The IMU modality has a strong systematic direction bias that could theoretically be informative, but the model cannot anchor it to the unstable skeleton. Combined with low motion diversity, the model has no stable person-specific features to learn, and embeddings collapse.
>
> **TC succeeds because both modalities provide stable, person-specific signals that can be aligned across diverse motions.**

---

## 7. Recommended Fixes (Priority Order)

| Priority | Fix | Expected Impact |
|----------|-----|-----------------|
| 1 | **Use accurate 3D skeleton** (e.g., multi-view cameras, MoCap, or reliable 3D pose estimator) | High — solves primary instability |
| 2 | **Standardize IMU sensor orientation** across persons | Medium-High — removes systematic bias |
| 3 | **Increase motion diversity** per session (require distinct actions) | Medium — provides more discriminative signal |
| 4 | **Minimize static periods** and remove nearly-static sessions | Medium — reduces uninformative data |
| 5 | **Use subjects with visibly different body types** | Low-Medium — secondary factor |

The single most important fix is **skeleton quality**. Without stable 3D skeleton, the alignment model cannot learn reliable identity representations regardless of IMU quality or motion diversity.
