# E1 Fine-Tune Fusion Result: seed0

## Setup
- Source: EgoHumans dual-embedding (Model-L + Model-G), seed 0
- Target: custom w24, both branches fine-tuned on custom train
- Fusion: `score = α·score_L + (1−α)·score_G`

## Result

| Setting | Best α | Mean FrameAcc |
|---|---|---|
| Zero-shot (source only) | 1.0 | 0.2940 |
| Fine-tuned Local only | 1.0 | 0.7259 |
| **Fine-tuned Local + Fine-tuned Global** | **0.9** | **0.7332** |

## Per-Clip Detail

| Clip | Local-only (α=1.0) | Fusion (α=0.9) | Δ |
|---|---|---|---|
| 171423_seg0 | 0.3975 | 0.4475 | **+0.050** |
| 171423_seg1 | 0.3320 | 0.3660 | **+0.034** |
| 171724_seg0 | 1.0000 | 1.0000 | 0.000 |
| 171724_seg1 | 1.0000 | 1.0000 | 0.000 |
| 172257_seg0 | 0.9459 | 0.9459 | 0.000 |
| 172522_seg0 | 0.6979 | 0.6811 | -0.017 |
| 172522_seg1 | 0.7079 | 0.6917 | -0.016 |

## Key Observations

1. **Dual fine-tuned fusion (0.7332) slightly outperforms fine-tuned local only (0.7259)**.
2. **Global branch becomes helpful after target-domain fine-tuning**: best α=0.9 (not 1.0), unlike zero-shot/source-only where α=1.0 was best.
3. **Global branch mainly helps on 171423** (~0.04 improvement), where local-only struggled.
4. **171724 remains perfect** with local-dominated fusion (α ≥ 0.7).
5. **Performance is very close to G4/E11 from-scratch** (0.7332 vs 0.752 ± 0.095), but still slightly below mean.

## Implications

- Pre-training on EgoHumans + fine-tuning on custom is a viable alternative to from-scratch training.
- The global branch needs target-domain fine-tuning to become useful; zero-shot global is harmful.
- Multi-seed evaluation is critical to determine whether 0.7332 is consistently below or comparable to from-scratch 0.752.
