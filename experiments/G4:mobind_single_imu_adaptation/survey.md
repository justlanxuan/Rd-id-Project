# G4 Survey: 单 IMU 匹配的现有基线与 MoBInd 现状

## 1. MoBInd 原始架构

- **论文/代码：** MoBInd: Motion Bidirectional Transformer for In-the-wild Motion Capture and Identity Association。
- **核心思想：** 两阶段对比学习 + Masked Autoencoder。
  - **Stage1**：instance-level 对比学习，拉近匹配的 `(IMU window, motion limb window)` 对。
  - **Stage2**：加载 Stage1 encoder，在 multi-sensor 模式下加入 MAE 重建目标，让 IMU 特征能重建被 mask 的 motion token。
- **默认设置：** EgoHumans 数据集使用 5 个肢体（`LeftWrist, RightWrist, LeftKnee, RightKnee, Head`），每个肢体对应一个 IMU。

## 2.  Autism-project 中已有的相关实验

### G_egohumans 控制变量实验（修正后）

| 实验 | IMU | 窗口 | FrameAcc | 关键结论 |
|---|---|---|---|---|
| E6-correct | 1 IMU / RightWrist | 24 帧 | **0.9548** | 单 IMU + 单肢体 motion 可行 |
| E8 | 1 IMU / RightWrist | 100 帧 | **0.9616** | 单 IMU 长窗口也强 |
| E9 | 5 IMU | 24 帧 | **0.9641** | 多 IMU 略好 |
| E7 | 5 IMU | 100 帧 | **0.9675** | 全设置上限 |

- **限制：** E6/E8 的 `num_limbs=1` 同时把视频侧 motion 也限制为 `RightWrist` 肢体，做的是“同关节 IMU↔pose”匹配，不是真正的“单 IMU + 全视频骨架”。

### G3/E2: MoBInd on Custom Same Split

| 设置 | seeds | mean FrameAcc | std |
|---|---|---|---|
| w24 | 0/42/123 | 0.782 | 0.086 |
| w100 | 0/42/123 | 0.696 | 0.172 |
| w24 | 1/2/3 | 0.564 | 0.200 |
| w100 | 1/2/3 | 0.621 | 0.094 |
| 6 seeds 合并 | 0/42/123/1/2/3 | ~0.67 | ~0.17 |

- **关键发现：** custom 上性能 **seed 方差极大**，说明单 IMU MoBInd 在当前数据上不稳定。
- **对照 SOTA：** Autism-project pipeline custom-only same split = **0.613 ± 0.010**。

## 3. 可借鉴方向

1. **Motion 侧信息补偿：** 单 IMU 缺失多肢体互补信息，可用完整骨架或学习骨架不确定性来补偿。
2. **MAE 目标改造：** 单 IMU 时无法做“跨肢体 mask 重建”，可改为跨时间 patch 重建或跨模态重建。
3. **数据增强：** 针对 IMU 的 SO(3) 旋转、噪声、 magnitude scaling 可能提升泛化。
4. **预训练 / 迁移：** 在 EgoHumans 上训练单 IMU encoder，再在 custom 上 fine-tune（E2 的 zero-shot 迁移失败，fine-tune 待验证）。
5. **架构轻量化/深度化：** 单 IMU 输入更简单，可能需要更深/更宽的时序 encoder 或注意力机制。

## 4. 当前空白

- 没有系统研究“单 IMU + 全骨架”在 MoBInd 中的可行性。
- 没有针对单 IMU 的 MAE 目标改造实验。
- 没有 IMU-specific 数据增强对 custom seed 稳定性的量化分析。
