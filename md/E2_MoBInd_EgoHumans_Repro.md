# E2: MoBInd official EgoHumans reproduction

## Overview
Independent reproduction of the MoBInd official baseline on the EgoHumans dataset,using the isolated conda environment `mobind_repro`.

## Environment
- GPU: NVIDIA RTX 4090 D
- CUDA: 11.8
- Python: 3.10
- PyTorch: 2.1.0+cu118
- NumPy: 1.26.4 (NumPy 2.x is incompatible with torch 2.1.0)
- MoBInd repo: `/home/fzliang/MoBind`
- Data: `/data/lyxie/ReID/Data/egohumans`
- Checkpoints: `/home/fzliang/MoBind/checkpoints/EgoHumans/stage1`, `stage2`

## Code fixes applied to MoBInd
A patch is saved at `third-party/mobind_egohumans_fixes.patch` and covers:
- `preprocess/EgoHumans/cache.py` and `cache_multi_person.py`: cast `numpy.int64` to Python `int` for JSON serialization.
- `builder/build_model.py`: remove imports of undefined modules and correct `ConvFormer` import path.
- `eval_sync_egoh.py`: support multi-person `(P, T, ...)` arrays and fix duplicated `gt_offsets.append`.

## Caches built
| Cache | Samples |
|-------|---------|
| `cache_action_5_2` | 4659 |
| `cache_action_multi_5_2` | 299 |
| `cache_sync_action_20_5` | 1540 |

## Results (stage2 MAE checkpoint)

### Retrieval
| Direction | R@1 | R@3 | R@5 | R@10 | R@25 | R@50 |
|-----------|-----|-----|-----|------|------|------|
| IMU → Video | 0.8264 | 0.9069 | 0.9289 | 0.9550 | 0.9728 | 0.9864 |
| Video → IMU | 0.8368 | 0.9059 | 0.9268 | 0.9529 | 0.9791 | 0.9874 |

### Localization
| Task | Accuracy |
|------|----------|
| Person localization (overall) | 98.01% |
| Person localization P2 / P3 / P4 | 100% / 100% / 96.96% |
| Limb localization (conditioned on correct person) | 89.22% |

### Synchronization
| Task | MAE (s) | Acc@0.1 | Acc@0.2 | Acc@0.5 |
|------|---------|---------|---------|---------|
| Person-level | 0.0421 | 0.6545 | 0.9925 | 0.9988 |
| Video-level | 0.0392 | 0.6448 | 1.0000 | 1.0000 |

## Artifacts
Full HAROS-style experiment sandbox:
`experiments/G_egohumans/E2:mobind_reproduce/`
- `plan.md`
- `progress.md`
- `test/test.md`
- `results/results.md`
- `results/metrics.json`
- `results/figures/`
- `scripts/A1_run_full_repro.sh`
- `scripts/B1_visualize_results.py`

## Raw logs
`experiments/G_egohumans/E2:mobind_reproduce/logs/`
- `eval_retrieval.log`
- `eval_localization.log`
- `eval_sync_person.log`
- `eval_sync_video.log`
