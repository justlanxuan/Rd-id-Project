# E1 Plan: 单 IMU MoBInd 基线复现与评估框架

## 科学问题

在 custom same-split 上，MoBInd 单 IMU from-scratch 的 baseline 性能与稳定性到底如何？

## 实验设计

直接复用并整理 G3/E2 的已有结果，但统一使用 **6 seeds（0/42/123/1/2/3）** 作为 G4 所有后续子实验的对照锚点。

| 子实验 | 内容 | 产物 |
|---|---|---|
| A1 | 汇总 w24/w100 6 seeds 结果 | `results/multi_seed_summary.json` |
| A2 | 绘制 seed 稳定性图、per-clip 均值/方差表 | `results/results.md` |
| A3 | 封装一键评估脚本 | `scripts/eval_all_seeds.sh` |

## 配置

- 单 RightWrist IMU，7 通道。
- Stage1: `multi_sensor=false`, `motion_type=wjoint`。
- Stage2: `multi_sensor=true`, `num_limbs=1`, `limb_list=[RightWrist]`。
- 窗口：w24（0.8s）、w100（3.333s）。
- seeds：0, 42, 123, 1, 2, 3。

## 评估指标

- FrameAcc mean ± std over 6 seeds。
- Per-clip mean ± std。
- 与 E10b pipeline SOTA（0.613 ± 0.010）对比。

## 预期结论

- 确认 baseline 均值与方差，为后续 I1–I8 改进提供对照。
- 如果 baseline 本身已稳定 >0.75，则重点转向迁移/部署；否则优先做架构/目标函数改进。
