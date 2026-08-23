# E5 Results：EgoHumans realistic orientation/domain

E5 test contract 已通过。前两轮 canonical-TC/canonical-Ego screen 均降级为诊断；主表只使用 Ego、TC、Custom 三者均为 MotionBERT/AlphaPose source-aligned 的 fully-aligned cache。七个无 AlphaPose cache 的 Ego canonical test sessions 只作 domain-shift diagnostic。

## 结果表

| source regime | orientation track / variant | Ego AlphaPose validation* | Custom23 high | Custom23 full | Custom23 low | 57/22/24 controls | status |
|---|---|---:|---:|---:|---:|---|---|
| EH-only | O0 baseline (3 seeds) | 0.430 | 0.494 | 0.470 | 0.439 | ≈0.50 | screen; fully aligned |
| EH-only | O3D `turning_cross` (3 seeds) | 0.343 | 0.500 | 0.493 | 0.485 | ≈0.50 | harms Ego validation |
| TC-only | O0 baseline (5 seeds) | 0.336 | 0.479 | 0.484 | 0.491 | ≈0.50 | confirmation control |
| TC-only | O3D `turning_cross` (5 seeds) | 0.332 | 0.482 | **0.504** | **0.532** | ≈0.50 | current best, weak gain |
| TC+EH balanced | O0 baseline (5 seeds) | 0.373 | 0.457 | 0.476 | 0.500 | ≈0.50 | mixed control |
| TC+EH balanced | O3D `turning_cross` (5 seeds) | 0.371 | 0.471 | 0.494 | 0.523 | ≈0.50 | no stable full gain |
| all regimes | O2D AlphaPose proxy (3 seeds) | — | ≈0.48–0.50 | ≈0.48–0.51 | — | ≈0.50 | no gain |

\* Ego column is session-disjoint AlphaPose validation, not a test set. Custom23 values are frozen test means; high/low denominators are 56/44 rows (28/22 groups), full is 100 rows (50 groups).

The fully-aligned 5-seed TC-only confirmation gives O3D `turning_cross` versus O0: high-turn 0.479→0.482 (+0.4 pp), low-turn 0.491→0.532 (+4.1 pp), full 0.484→0.504 (+2.0 pp). The mixed 5-seed confirmation is similarly small (full 0.476→0.494). These are diagnostic candidate results, not an E4.1 freeze replacement.

## 参数与 provenance

主要 artifacts：split `/data/fzliang/reid-project/g12/e5_egohumans_orientation/manifests/egohumans_e5_session_split.json`；source-aligned manifests `.../e4_1_source_aligned/motionbert_alphapose_cache_v2/manifests/e5_ego_{train,validation}.csv`；3D screen `.../screen_source_aligned/`；2D control `.../screen_source_aligned_o2d/`；5-seed confirmation `.../confirmation_source_aligned/`；summary `.../e5_screen_summary.json`。

## AI 独立反思

当前证据不支持在 fully-aligned 自有 extractor 轨上宣称稳定的朝向增益：TC-only O3D cross 仅约 +2.0 pp full，Ego validation 反而下降；TC+EH balanced 也没有稳定优于 baseline；O2D AlphaPose proxy 无增益。当前可选的最佳训练配置是 TC-only + O3D `turning_cross`，但它仍接近 chance，且明显弱于已有 E4.1 physical-turning-MoE freeze，不能替代后者。

## 人机讨论纪要

- 2026-08-22：用户要求在 EgoHumans realistic 上验证朝向作用，并寻找 Ego-only、TC-only 与双 source 共同训练的最佳方式；按 HAROS 新开 E5。
