# E1 Fine-Tune Result: Local Branch, seed0

## Setup
- Source: EgoHumans Model-L (local), seed 0
- Target: custom w24, full fine-tune Stage2 on custom train
- Evaluated on custom test clips

## Result

| Norm | Best α | Mean FrameAcc |
|---|---|---|
| none | **1.0** | **0.7259** |

## Per-Clip Detail (α=1.0, pure local fine-tuned)

| Clip | FrameAcc |
|---|---|
| 171423_seg0 | 0.3975 |
| 171423_seg1 | 0.3320 |
| 171724_seg0 | **1.0000** |
| 171724_seg1 | **1.0000** |
| 172257_seg0 | 0.9459 |
| 172522_seg0 | 0.6979 |
| 172522_seg1 | 0.7079 |

## Key Observations

1. **Huge improvement over zero-shot**: 0.2940 → 0.7259 (+0.432).
2. **Close to G4/E11 from-scratch**: 0.7259 vs 0.752 ± 0.095 (single seed already near mean).
3. **Global branch (source) is still harmful**: best α=1.0, meaning fine-tuned local alone is better than any fusion with source global.
4. **171724 (previously complete zero-shot failure) becomes perfect after fine-tune**.
5. **171423 remains weak**: both seg0/seg1 are below 0.40, suggesting this session has unique domain characteristics.

## Implications

- EgoHumans pre-training provides a useful initialization for custom, but only after full fine-tuning.
- The global branch learned on EgoHumans does not transfer; possibly training only the local branch on EgoHumans would be more efficient.
- Multi-seed evaluation is needed to confirm stability vs G4/E11 from-scratch.
