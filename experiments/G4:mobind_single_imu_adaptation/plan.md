# G4 Plan: MoBInd 单 IMU 适配探索路线图

## 总体策略

采用 **控制变量 + 快速迭代** 的方式，从易到难验证 I1–I8 的改进假设。每个子实验都必须在 custom same-split 上跑 **≥6 seeds**，并以 E2 的 from-scratch 结果作为对照基线。

## Phase 1: 基线复现与评估框架（E1）

### E1:baseline_single_imu_mobind

- **目标：** 复现并扩展 G3/E2 的单 IMU from-scratch 基线，建立统一的评估脚本与 SOTA 锚点。
- **内容：**
  - A1: 复现 w24/w100 from-scratch，seeds 0/42/123/1/2/3（已有结果，整理进 E1）。
  - A2: 统一评估脚本，支持一键生成 `multi_seed_summary.json` 与 `results.md`。
  - A3: 绘制 seed 稳定性图、per-clip 热力图。
- **预期产出：** `E1:baseline_single_imu_mobind/results/results.md`。

## Phase 2: Motion 侧信息补偿（I1 / I2）

### E2:single_imu_full_skeleton

- **目标：** 验证“单 IMU + 完整 17 关节骨架”是否优于“单 IMU + 单肢体”。
- **变量：**
  - Stage1/Stage2 `motion_type` 从 `wjoint` 改为 `pose2d`。
  - motion_encoder input_channels 从 6 改为 34（17 joints × 2D）。
- **对照：** E1 单肢体 baseline。

### E3:joint_attention_single_imu

- **目标：** 在全骨架基础上引入关节注意力，让模型自动选择与 IMU 相关关节。
- **变量：** 在 motion encoder 前加入轻量关节注意力 / pooling。
- **对照：** E2 全骨架、E1 单肢体。

## Phase 3: MAE 目标改造（I3）

### E4:single_imu_temporal_mae

- **目标：** 把 Stage2 的肢体级 mask 改为时间 patch mask，适应单 IMU。
- **变量：** 修改 `ContrastiveMAE` 的 mask 策略，仅 mask IMU 的时间 patch。
- **对照：** E1 baseline MAE。

### E5:cross_modal_mae

- **目标：** 跨模态 mask：用 motion 重建 masked IMU patch，强化单 IMU–全骨架对齐。
- **变量：** 双向 mask 与重建。
- **对照：** E4 时间 MAE、E1 baseline。

## Phase 4: 数据增强与预训练（I4 / I5）

### E6:single_imu_augmentation

- **目标：** 量化 IMU 数据增强对 seed 稳定性的影响。
- **变量：** SO(3) 旋转、噪声、scale、时间抖动等。
- **对照：** E1 无增强。

### E7:egohumans_pretrained_finetune

- **目标：** 用 E8/E6 单 IMU EgoHumans 检查点在 custom 上 fine-tune。
- **变量：** frozen Stage1 / full fine-tune / adapter。
- **对照：** E1 from-scratch、E2 zero-shot。

## Phase 5: 架构与训练策略（I6 / I7 / I8）

### E8:multi_scale_window

- **目标：** 多尺度窗口训练/推理。
- **变量：** 24/50/100 帧模型融合。

### E9:training_stability_sweep

- **目标：** 调优 Stage1/Stage2 训练超参降低 seed 方差。
- **变量：** patience、lr、batch size、warmup。

### E10:encoder_variant

- **目标：** 尝试替代 IMU encoder（TCN / PatchTST / TimesNet）。
- **变量：** 替换 `ConvFormer`。
- **对照：** E1 ConvFormer。

## Phase 6: 局部 + 整体联合对齐（I9 / I10）

### E11:dual_embedding_local_global

- **目标：** 验证 I9：训练两个 IMU embedding（local + global），推理时融合。
- **变量：**
  - local target：RightWrist 单肢体 motion。
  - global target：完整 17 关节 pose2d 或全肢体聚合。
  - 融合策略：平均、置信度加权、学习门控。
- **对照：** E1 local-only、E2 global-only、E11 融合版本。

### E12:curriculum_local_to_global

- **目标：** 验证 I10：课程式训练 local → global 或 global → local。
- **变量：**
  - Stage A 目标：local / global。
  - Stage B 目标：global / local。
  - 是否冻结 encoder、学习率比例。
- **对照：** E11 同时训练、E1 baseline。

### E13:adaptive_local_global_gating

- **目标：** 验证 I11：根据肢体运动丰富度自适应选择 local 或 global 分支。
- **变量：**
  - 丰富度指标：IMU 能量、关节速度方差、可见性分数等。
  - 门控方式：硬路由 vs 软门控（可学习或规则）。
- **对照：** E11 静态融合、E1 local-only、E2 global-only。

## 最终阶段：SOTA 锚定与报告

- 汇总所有子实验，选择最佳组合。
- 更新 `experiments/SOTA_reproduce.md`。
- 撰写 G4 最终报告。

## 风险控制

| 风险 | 应对 |
|---|---|
| 修改 MoBInd 源码引入 bug | 每次改动前备份，先在 toy 数据或 1 seed 上验证 |
| 训练耗时不可控 | 优先在 w24 上做快速消融，有效后再扩展到 w100 |
| seed 方差大导致结论不稳定 | 所有子实验必须 ≥6 seeds，用均值+std 判断 |
| motion_type 改变需重建 cache | 为每个 motion_type 单独建 cache 目录 |
