# G5 Ideas: Cross-Dataset Transfer 改进假设

> 本文件按追加式记录；验证后可更新状态（未验证 / 已验证有效 / 已验证无效）。

---

## I1: EgoHumans 预训练 dual-embedding，custom fine-tune

- **描述：** 将 G4/E11 的 local + global dual-embedding 架构先在 EgoHumans 上完整训练（Stage1+Stage2），然后将两个 branch 的 checkpoint 加载到 custom 上，进行 fine-tune。
- **验证：** E1 的核心实验。对比：
  - 源域 only（EgoHumans test 性能）
  - target zero-shot（custom test，不训练）
  - target fine-tune（custom train → custom test）
  - target from-scratch（G4/E11 基线）
- **风险：** EgoHumans 与 custom 域差距大，fine-tune 可能仍无法超过 from-scratch。

## I2: 仅预训练 Local branch 或 Global branch

- **描述：** 分别验证 Local branch（RightWrist ↔ RightWrist motion）和 Global branch（RightWrist ↔ full pose2d）的迁移能力。可能 Local branch 因任务更简单而迁移更稳定。
- **验证：** 在 EgoHumans 上分别训练单 branch 模型，再在 custom 上 fine-tune，与 dual-embedding 对比。
- **风险：** 单 branch 在 custom 上已验证不如 fusion，迁移后可能仍不如 dual。

## I3: 跨域 IMU 统计归一化

- **描述：** EgoHumans 是 synthetic IMU，custom 是 real IMU，加速度/角速度分布可能差异显著。在训练前对 IMU 做 per-dataset 标准化（z-score）或 learnable domain-invariant normalization。
- **验证：** 比较无归一化、per-dataset z-score、Domain-Adversarial Training（DANN/Gradient Reversal）对迁移的影响。
- **风险：** 若域差距主要来自动作分布而非传感器统计，归一化收益有限。

## I4: 渐进式 fine-tune（progressive unfreezing）

- **描述：** 先冻结 IMU encoder 和 motion encoder，只训练 fusion / projection head；然后逐步解冻低层/高层，避免破坏源域对齐表示。
- **验证：** 设计多阶段 fine-tune schedule，对比一次性 full fine-tune。
- **风险：** schedule 复杂，可能过拟合到 custom 小数据集。

## I5: Adapter-based domain adaptation

- **描述：** 在 IMU encoder 和 motion encoder 后各加轻量 adapter（如 FiLM、bottleneck MLP），源域训练时禁用，目标域 fine-tune 时只更新 adapter。
- **验证：** 对比 frozen encoder + adapter、full fine-tune、from-scratch。
- **风险：** E9 中 frozen adapter 无效，可能 MoBInd 的表示空间需要更大调整。

## I6: 联合预训练 EgoHumans + custom

- **描述：** 不先做源域预训练再迁移，而是直接在 EgoHumans + custom 的合并数据上训练 dual-embedding 模型，然后评估 custom。
- **验证：** 与 I1 的两阶段方法对比。
- **风险：** 数据分布不平衡，EgoHumans 样本远多于 custom，模型可能偏向源域。

## I7: 多窗口融合 + 迁移

- **描述：** 在 EgoHumans 上分别预训练 w24 和 w100 dual-embedding 模型，迁移到 custom 后做多窗口 late fusion。
- **验证：** 对比单窗口与多窗口融合在迁移 setting 下的效果。
- **风险：** 训练/推理成本翻倍。

## I8: 基于 confidence 的自适应源域选择

- **描述：** 若同时有多个源数据集（EgoHumans、TotalCapture），根据目标样本与源域的相似度动态选择或加权源域模型。
- **验证：** 构建 source model ensemble，按 target 样本 confidence 加权。
- **风险：** 当前仅 EgoHumans 一个源数据集，I8 为远期方向。
