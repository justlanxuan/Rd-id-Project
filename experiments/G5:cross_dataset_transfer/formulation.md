# G5 Formulation: Cross-Dataset Transfer for Single-IMU Person Re-ID

## 1. 当前挑战 (Need)

- custom 数据集规模小、场景单一（室内儿童互动），从头训练单 IMU 模型时 **seed 方差大**、性能不稳定。
- EgoHumans 是公开的大规模 egocentric 多人数据集，包含丰富的身体姿态与 IMU-motion 对齐样本，但直接 zero-shot 迁移到 custom 效果很差（G_egohumans/E9: zero-shot 0.339，frozen adapter 无效）。
- G4/E11 在 custom 上验证了 **local + global 双分支 embedding 融合** 的架构潜力（w24 Fusion best α = 0.752 ± 0.095），但该架构仅在 custom 上从头训练，尚未利用更大规模的源数据集。
- 核心问题：**如何在源数据集（EgoHumans）上预训练 dual-embedding 表示，使其在 custom 上通过 zero-shot 或少量 fine-tune 获得稳定提升？**

## 2. 实验目标 (Goal)

在保持单 IMU `RightWrist` 输入的前提下，系统探索跨数据集迁移学习：

1. **源域预训练**：在 EgoHumans 上训练 local/global dual-embedding 模型，学习通用的单 IMU ↔ 姿态对齐表示。
2. **目标域迁移**：将预训练模型迁移到 custom，对比：
   - **Zero-shot**：直接评估，不做任何目标域训练。
   - **Fine-tune**：加载源域权重，在 custom 上继续训练（full fine-tune / partial freeze / adapter）。
3. **对照基线**：custom 上 from-scratch 训练的 dual-embedding 模型（G4/E11）。
4. **明确迁移价值**：量化 EgoHumans 预训练对 custom 性能、seed 稳定性的影响。

## 3. 监测指标与达标线 (Metrics)

| 指标 | 来源 | 说明 |
|---|---|---|
| FrameAcc | MoBInd `eval_synchronous` | 帧级身份匹配准确率，主指标 |
| Seed mean ± std | 多 seed 聚合 | 评估训练稳定性 |
| Per-clip FrameAcc | 按 clip 分解 | 定位域差异大的 session |
| Val top1 | MoBInd 训练日志 | 窗口级 IMU→motion R@1 |
| Zero-shot vs Fine-tune Δ | 相对源域基线 | 量化迁移收益 |

**成功标准：**
- EgoHumans 预训练 + custom fine-tune 的 FrameAcc 在 6 seeds 上**稳定超过** custom from-scratch（0.752 ± 0.095）。
- 或：zero-shot 性能显著优于历史 E9 zero-shot（0.339），证明 dual-embedding 表示具有更好的跨域泛化性。
- 明确哪种迁移策略（full fine-tune / Stage1-only freeze / adapter / progressive unfreezing）最有效。

## 4. 与现有实验的关系

- **前置依赖：**
  - G_egohumans/E6–E8：MoBInd 在 EgoHumans 上的单 IMU 训练方法与检查点。
  - G3/E2：custom same-split 数据准备与 local model（RightWrist↔RightWrist）训练。
  - G4/E11：custom 上 dual-embedding（local + global）的 from-scratch 结果与融合策略。
- **对照基线：**
  - G4/E11 w24 from-scratch：0.752 ± 0.095（Fusion best α）。
  - G_egohumans/E9：E8 single-IMU → custom zero-shot/finetune（旧 Autism pipeline）。
  - G_egohumans/E10：EgoHumans + custom 联合训练（旧 Autism pipeline）。
- **目标 SOTA：** G4/E11 在 custom same-split 上保持的 0.752 ± 0.095。

## 5. 研究问题（待逐步验证）

1. EgoHumans 上预训练的 dual-embedding 模型，在 custom 上 zero-shot 能否超过历史 E9 zero-shot？
2. 哪种 fine-tune 策略最有效：全量微调、只微调 Stage2、只训练 adapter、渐进解冻？
3. Local branch 和 Global branch 的迁移能力是否不同？（源域 EgoHumans 上 global 可能更强，目标域 custom 上 local 可能更稳定）
4. 是否需要对 IMU 信号做跨数据集归一化（如 domain-invariant representation、IMU statistics alignment）？
5. EgoHumans 与 custom 的域差距主要来自哪些方面（传感器 mounting、动作分布、相机视角、人数）？
