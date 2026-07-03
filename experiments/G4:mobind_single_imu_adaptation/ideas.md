# G4 Ideas: 单 IMU 适配的改进假设

> 本文件按追加式记录；验证后可更新状态（未验证 / 已验证有效 / 已验证无效）。

---

## I1: Motion 侧使用完整骨架而非单肢体

- **描述：** 当前单 IMU MoBInd 把 motion 侧也限制为同一个肢体（RightWrist），任务被简化。尝试让单 IMU 与完整 17 关节骨架（`pose2d` motion_type）匹配，用更丰富的空间信息补偿 IMU 数量不足。
- **验证：** 修改 Stage1/Stage2 的 `motion_type` 为 `pose2d`，保持单 IMU 输入，评估性能/稳定性变化。
- **风险：** motion 维度变大（17×2=34），可能需要调整 encoder 或训练预算。

## I2: 全骨架 + 关节级注意力 / 软对齐

- **描述：** 直接 pose2d 输入维度过高且包含无关关节。引入关节级注意力或 learnable mask，让模型自动关注与 RightWrist IMU 最相关的关节子集。
- **验证：** 在 motion encoder 前加入关节注意力模块，对比固定单肢体与全骨架。

## I3: 改造 MAE 目标以适应单 IMU

- **描述：** Stage2 原 MAE 通过 mask 掉部分肢体来重建。单 IMU 时只有 1 个肢体，可改为：
  - 时间维度 mask：随机 mask IMU 的部分时间 patch，重建对应 motion patch。
  - 跨模态 mask：mask IMU patch，用 motion 侧信息指导重建，强化 IMU–motion 对齐。
- **验证：** 修改 `ContrastiveMAE` 的 mask/reconstruct 逻辑，比较不同 mask 策略。

## I4: 单 IMU 专用数据增强

- **描述：** custom 数据量小，seed 方差大。对单 IMU 7 维信号做随机 SO(3) 旋转、高斯噪声、scale jitter、时间抖动等增强。
- **验证：** 在 Stage1/Stage2 训练时加入 IMU augmentation，跑多 seed 看 std 是否下降。

## I5: 在 EgoHumans 上预训练单 IMU encoder

- **描述：** E2 显示 EgoHumans→custom zero-shot 差，但 fine-tune 可能有效。用 E6/E8 的 Stage1/Stage2 作为预训练权重，在 custom 上 fine-tune。
- **验证：** 加载 E8 Stage2 权重，在 custom 上 fine-tune 全部或部分层。

## I6: 更大的时序上下文 / 多尺度窗口

- **描述：** 单 IMU 信息少，可能需要更长或可变窗口。尝试多尺度窗口训练/推理（24/50/100 帧融合）。
- **验证：** 训练多个窗口模型，推理时做 late fusion 或窗口投票。

## I7: 更稳定的训练策略

- **描述：** 高 seed 方差可能来自 early stopping、学习率、batch size。系统调优这些超参。
- **验证：** 对 Stage1/Stage2 的 patience、lr、batch size 做控制变量。

## I8: 针对单 IMU 的架构变体

- **描述：** 尝试替换 ConvFormer 为更擅长单变量时序的 encoder（如 TCN、TimesNet、PatchTST），或在 IMU encoder 加入跨窗口注意力。
- **验证：** 保持 MoBInd 两阶段框架，替换 encoder，评估 FrameAcc 与效率。

## I9: 双 IMU Embedding — 局部语义 + 整体语义共同决策

- **描述：** 单 IMU 既包含局部肢体运动信息（如手腕抖动），也间接反映整体运动（如走路时全身节律）。训练**两个 IMU embedding 分支**：
  - **Local branch**：与同一肢体 motion（RightWrist）对齐。
  - **Global branch**：与完整 17 关节骨架（pose2d 或全肢体聚合）对齐。
  推理时对两个分支的匹配结果做融合，可为每个分支设置置信度（如基于该窗口运动丰富度、IMU 能量、或学习一个 gating 网络）。
- **实现思路：**
  - 方案 A：两个独立的 IMU encoder（local / global），共享或不共享 backbone。
  - 方案 B：一个共享 IMU encoder + 两个投影头。
  - 融合方式：softmax 加权、可学习门控、基于方差/熵的置信度加权。
- **验证：** 对比单分支 local-only、单分支 global-only、双分支融合三种设置。
- **风险：** 参数量翻倍；global branch 可能因输入维度高而更难训练。

## I10: 两阶段课程训练 — 先局部后整体（或先整体后局部）

- **描述：** 用课程学习的方式分阶段训练单 IMU 表示：
  - **Local → Global**：先用局部肢体对齐学到稳定的基础表示，再扩展到完整骨架对齐。
  - **Global → Local**：先用完整骨架约束全局运动语义，再细化到局部肢体，强化手腕等局部细节。
- **实现思路：**
  - Stage A：固定 motion 为单肢体，训练 IMU encoder。
  - Stage B：切换 motion 为完整骨架，冻结或低学习率微调 encoder，继续训练。
  - 也可在同一网络中交替使用两种监督（每个 epoch 或每个 batch 随机选择 local/global target）。
- **验证：** 与同时训练 local+global 的 I9 做对比；比较训练稳定性与最终性能。
- **与 I9 的关系：** I9 是“同时学习两个表示再融合”，I10 是“分先后学习同一个/多个表示”。两者可结合：I9 的 dual-branch 也可分别用 I10 的课程策略预训练。

## I11: 基于肢体活动丰富度的自适应局部/整体匹配

- **核心假设：** 单 IMU 佩戴肢体的运动状态决定了“局部肢体匹配”与“整体动作匹配”哪个更可靠：
  - **肢体运动丰富时**（如挥手、操作物体）：RightWrist IMU 与 RightWrist motion 高度相关，局部匹配更准。
  - **肢体运动贫乏/被遮挡时**（如手插兜、静止、遮挡）：局部信号噪声大甚至缺失，整体动作（走路节律、身体摆动）更稳定。
- **实现思路：**
  1. **训练数据分类**：对每个窗口定义“肢体丰富度”指标，如：
     - RightWrist 关节在窗口内的 2D 位移方差 / 速度能量。
     - RightWrist IMU 的加速度/角速度能量。
     - AlphaPose 该关节的可见性分数。
     按阈值把训练样本分为 **rich-limb** 和 **poor-limb** 两类。
  2. **自适应匹配策略：**
     - **硬路由（hard routing）**：rich 窗口只用 local branch，poor 窗口只用 global branch。
     - **软门控（soft gating）**：训练一个轻量网络（或基于规则的置信度）预测 `p_local`，最终匹配分数 = `p_local * score_local + (1 - p_local) * score_global`。
  3. **与 I9/I10 的关系：** I9 是静态双分支融合，I11 是**按样本动态选择**哪个分支更可靠；I10 是课程式训练，可与 I11 结合：先分别训练两个分支，再用门控网络学习如何组合。
- **验证：**
  - 对比：local-only、global-only、I9 静态平均、I11 自适应门控。
  - 关键指标：不仅看整体 FrameAcc，还要看 rich/poor 子集上的性能提升。
- **风险：**
  - 需要定义合理的“丰富度”指标；若指标与身份相关，可能泄露标签。
  - 硬路由可能导致训练样本不平衡（poor 窗口可能远多于 rich）。
