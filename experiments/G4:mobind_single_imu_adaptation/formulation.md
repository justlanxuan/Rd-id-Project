# G4 Formulation: MoBInd 单 IMU 适配与最优方案探索

## 1. 当前挑战 (Need)

- MoBInd 原生设计面向 **多 IMU / 多肢体** 场景（EgoHumans 默认 5 个肢体：左右手腕、左右膝盖、头）。其 Stage2 的 `ContrastiveMAE` 通过 `num_limbs` 个肢体的 local feature 聚合获得 global embedding。
- 将 MoBInd 压缩到 **单 IMU** 后，Stage2 的跨肢体聚合头退化为 `S=1` 的线性投影，模型实际上退化为“单 IMU ↔ 单肢体 motion”的匹配器。
- 在 custom 数据上的初步实验（G3/E2）显示：单 IMU MoBInd 虽然能超过 Autism-project pipeline SOTA（0.613），但 **seed 方差极大**（w24 在 seeds 0/42/123 上 0.782 ± 0.086，在 seeds 1/2/3 上 0.564 ± 0.200）。
- 当前并不清楚：**单 IMU 的性能波动是来自数据量小，还是 MoBInd 的多肢体架构在单 IMU 设置下存在结构性不适配**。

## 2. 实验目标 (Goal)

在 **custom 真实 IMU + AlphaPose 骨架** 的 same-split 设置下，系统探索如何让 MoBInd **专为单 IMU 设计或适配**，实现：

1. **稳定的高性能**：在 6+ seeds 上取得 mean FrameAcc ≥ 0.75 且 std ≤ 0.05。
2. **真正“单 IMU + 全视频骨架”**：不再把 motion 侧限制为同一个肢体，允许单 IMU 与完整 17 关节骨架匹配。
3. **可解释性**：明确哪些改动对单 IMU 场景有效（架构、目标函数、数据增强、预训练）。

## 3. 监测指标与达标线 (Metrics)

| 指标 | 来源 | 说明 |
|---|---|---|
| FrameAcc | `eval_synchronous` | 帧级身份匹配准确率，主指标 |
| Seed std | 多 seed 聚合 | 评估训练稳定性 |
| Per-clip mean/std | `multi_seed_summary.json` | 定位困难 clip |
| Val top1 | MoBInd 训练日志 | 窗口级 IMU→motion R@1 |
| 消融对比 | 相对 baseline Δ | 每项改动带来的提升 |

**成功标准：**
- 至少找到 1–2 项针对单 IMU 的有效改进，使 custom same-split FrameAcc 在 6 seeds 上稳定达到 **0.75+**。
- 明确单 IMU 场景下 MoBInd 的瓶颈是“数据量”还是“架构目标函数”。

## 4. 与现有实验的关系

- **前置依赖：** G3/E2（MoBInd on custom same split）、G_egohumans/E6–E9（MoBInd 单/多 IMU 控制变量）。
- **对照基线：** G3/E2 的 w24/w100 from-scratch 结果（seeds 0/42/123 与 1/2/3）。
- **目标 SOTA：** E10b pipeline = 0.613 ± 0.010。

## 5. 研究问题（待逐步验证）

1. 单 IMU 时，把 motion 侧从“单肢体”扩展到“全骨架”是否能提升性能/稳定性？
2. MoBInd Stage2 的 MAE 目标在单 IMU 下是否仍然有效？是否需要改为跨时间/跨模态 mask？
3. 针对单 IMU 设计更深的时序 encoder 或更大的 patch 是否有帮助？
4. 数据增强（IMU 旋转、噪声、scale）能否降低 seed 方差？
5. 预训练策略：在 EgoHumans 上预训练单 IMU encoder 后再 fine-tune custom，是否能缩小域 gap？
