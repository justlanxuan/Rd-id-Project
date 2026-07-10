# G5 Formulation: Cross-Dataset Transfer for Single-IMU Person Re-ID

## 1. 当前挑战 (Need)

- custom 数据集规模小、场景单一（室内儿童互动），从头训练单 IMU 模型时 **seed 方差大**、性能不稳定。
- EgoHumans 是公开的大规模 egocentric 多人数据集，但其 MoBind-like 合成 IMU 与 custom 真实 IMU 之间存在显著的 **domain gap**：直接 zero-shot 迁移到 custom 效果很差（G5/E1 EgoHumans source dual-embedding 仅 0.2940）。
- G4/E11 在 custom 上验证了 **local + global 双分支 embedding 融合** 的架构潜力（w24 Fusion best α = 0.752 ± 0.095），但该架构仅在 custom 上从头训练，尚未充分利用更大规模且域对齐的源数据集。
- E4b 进一步发现：使用 **realistic 合成 IMU** 作为源域（single-source direct transfer）可以显著降低域差距，zero-shot 4-source-seed 为 **0.5835 ± 0.1300**（其中 seed0 单点为 **0.7077**），fine-tune 为 **0.6928 ± 0.0604**；同设置下 MoBind-like 单源明显较弱（zero-shot **0.4495**，fine-tune **0.5653 ± 0.0113**）。
- G5/E2 已验证：realistic dual-embedding source 的 zero-shot 明显优于 EgoHumans source，但 full fine-tune 后 multi-seed 均值与方差不如 single-source conservative transfer。
- G5/E3 已验证：realistic single-source + low-LR target fine-tune 达到 **0.7413 ± 0.0241**，但该实验使用 RightWrist motion target。custom IMU 实际佩戴在 LeftWrist，因此 E3 现在只作为 historical mislabeled-wrist baseline。
- 核心问题已更新为：**如何在左手腕单 IMU 场景下保留 realistic source 的跨域表示，同时避免目标域小数据 fine-tune 破坏表示，并进一步用 fusion/canonicalization/temporal logic 提升泛化？**

## 2. 实验目标 (Goal)

在保持单 IMU `LeftWrist` 输入/运动目标的前提下，系统探索跨数据集迁移学习：

1. **源域预训练**：在 realistic 合成 IMU 上训练 single-source 与 local/global dual-embedding 模型，学习 IMU ↔ pose2d 对齐表示。
2. **目标域迁移**：将预训练模型迁移到 custom，对比：
   - **Zero-shot**：直接评估，不做任何目标域训练。
   - **Fine-tune**：加载源域 Stage2 权重，在 custom 上继续训练 Stage2。
   - **Conservative fine-tune**：降低学习率或控制可训练参数，避免破坏 source representation。
3. **对照基线**：
   - G5/E1：EgoHumans source dual-embedding（seed0）。
   - E4b：realistic 与 MoBind-like single-source direct transfer（realistic 4 source seeds zero-shot / 3 seeds fine-tune；MoBind-like seed0 zero-shot / 3 seeds fine-tune）。
   - G4/E11：custom from-scratch dual-embedding。
4. **明确迁移价值**：量化 realistic source、conservative target adaptation、dual fusion 对 custom 性能与 seed 稳定性的影响。

## 3. 监测指标与达标线 (Metrics)

| 指标 | 来源 | 说明 |
|---|---|---|
| FrameAcc | MoBInd `eval_synchronous` | 帧级身份匹配准确率，主指标 |
| Seed mean ± std | 多 seed 聚合 | 评估训练稳定性 |
| Per-clip FrameAcc | 按 clip 分解 | 定位域差异大的 session |
| Val top1 | MoBInd 训练日志 | 窗口级 IMU→motion R@1 |
| Zero-shot vs Fine-tune Δ | 相对基线 | 量化迁移收益 |

**成功标准：**
- Transfer 方法在 6 seeds 上达到或超过 `0.72` mean 且 std 不高于 `0.08`。
- Zero-shot 性能显著优于 G5/E1 zero-shot（0.2940），最好接近或超过 E4b realistic single-source zero-shot（0.5835 ± 0.1300）。
- 明确 dual-embedding 结构相比 single-source 是否有额外收益，以及何时会产生负迁移。

## 4. 与现有实验的关系

- **前置依赖：**
  - G4/E11：custom 上 dual-embedding（local + global）的 from-scratch 结果与融合策略（A5_eval_fusion.py）。
  - G3/E2：custom same-split 数据准备与 local model 训练。
  - G5/E1：EgoHumans source dual-embedding，用于对比源域选择。
  - E4b：realistic single-source direct transfer 基线。
- **对照基线：**
  - G4/E11 w24 from-scratch：0.752 ± 0.095（Fusion best α）。
  - G5/E1 w24 EgoHumans source dual-embedding：zero-shot 0.2940，fine-tune fusion 0.7332（seed0）。
  - E4b realistic single-source：zero-shot 0.5835 ± 0.1300（seed0: 0.7077），fine-tune 0.6928 ± 0.0604。
  - E4b MoBind-like single-source：zero-shot 0.4495，fine-tune 0.5653 ± 0.0113。
- **历史最佳 transfer：** G5/E3 realistic single-source low-LR fine-tune，0.7413 ± 0.0241；由于 wrist side 修正，该结果已由 E9 LeftWrist revalidation 替换为 corrected baseline。
- **当前 corrected LeftWrist 结果：** E9 realistic LeftWrist source -> custom zero-shot 0.6115 ± 0.1409，low-LR fine-tune 0.6148 ± 0.1638；direct custom LeftWrist from-scratch 0.6346 ± 0.1681。
- **当前 motion-side 诊断：** E11-A1 shoulder-anchored traditional matcher 达到 0.6352，说明相对骨架/角度/全局上下文信号可转化为 matching；E11-A3 50D kinematic state 走 MoBind `motion_type: feature` 只有 0.3355，说明瓶颈在模型接口/匹配目标；E11-A10 hybrid signed-window / abs-global consistency selector 达到 0.9014，首次超过 raw-keypoint direct seed0 0.8396；E11-A11 frozen-rule validation 进一步确认其超过 available direct-custom multi-seed baseline。
- **当前 SOTA on custom same-split：** G4/E11 0.752 ± 0.095。

## 5. 研究问题（逐步验证中）

1. Realistic 源域 dual-embedding 在 custom 上 zero-shot 是否优于 EgoHumans 源？是否优于 E4b single-source？
2. Full fine-tune 后 dual-embedding 融合是否优于 single-source fine-tune？当前答案：否。
3. Local branch 与 Global branch 的迁移能力是否不同？（源域上 global 通常更强；目标域上 local 更稳定）
4. 多 seed 稳定性是否优于 single-source 直接迁移？当前答案：E3 low-LR single-source 最稳定。
5. 跨域融合时的相似度归一化（none / zscore / minmax）如何选择？

## 6. 当前进展

- **G5/E1 EgoHumans source dual-embedding**：已完成 seed0，zero-shot 0.2940，fine-tune fusion 0.7332。
- **G5/E2 Realistic-IMU source dual-embedding**：已完成 6 seeds 的 cache、source 训练、zero-shot、fine-tune 与融合评估。详见 `E2:realistic_dual_embedding_pretrain/results/results.md`。
- **G5/E3 Realistic single-source conservative fine-tune**：已完成 6 seeds low-LR full fine-tune，0.7413 ± 0.0241；当前标记为 historical mislabeled-wrist baseline。
- **G5/E4 Realistic dual conservative fine-tune**：已完成 seed123 诊断；low-LR 只能修到 0.4894，暂不扩展。
- **G5/E5 Adaptive gate oracle diagnostics**：已完成。Direct custom dual 有 adaptive headroom，但 G5/E2 transfer dual 的 clip-level oracle 仍低于 E3。
- **G5/E6 IMU format audit + target jitter**：已完成。E4b/E3 20 Hz cache 文件名为 RightWrist，但 custom 真实佩戴为 LeftWrist；TotalCapture 48D 不应直接混入左手腕单 IMU 设置。Target train-only light acc jitter 在 seeds 0/42/123 上与 E3 low-LR 完全持平，但该结论也需视为 mislabeled-wrist context 下的历史结果。
- **G5/E9 LeftWrist revalidation**：已完成 explicit LeftWrist realistic source cache、custom LeftWrist alias cache、source pretrain、custom zero-shot/fine-tune 和 direct custom LeftWrist matched seeds 0/42/123。Corrected transfer mean 为 0.6148 ± 0.1638，低于 direct custom LeftWrist 0.6346 ± 0.1681。
- **G5/E10 Skeleton differential diagnostics**：已完成。Filtered bone geometry、elbow included angle、relative/global motion 与 IMU dynamics 有相关信号；但低维 replacement-feature learned model 失败，`hybrid_v1=0.5330` 低于 raw-keypoint direct seed0 0.8396。
- **G5/E11 Shoulder-anchored kinematic matching**：已完成 A1-A11。No-training traditional matcher 为 0.6352；50D `shoulder_kinematic_v1` MoBind feature-path seed0 为 0.3355；clip-global temporal assignment 为 0.6831；A10 hybrid consistency selector 为 0.9014；A11 confirms frozen A10 rule beats direct custom seeds 0/42/123 and 3-seed mean. 下一步若继续，应在新 held-out recording 上复验或实现无需 test-sweep 调参的 selector。
- **G5/E12 Realistic source-trained kinematic selector**：已完成 6 source folds。EgoHumans realistic source 训练稳定选择 `abs_global`，custom `0.6831 ± 0.0000`。它超过 MoBind-like fine-tune、E9 corrected transfer 和 E9 direct custom 3-seed mean，但低于 custom-selected A10 `0.9014`。
- **G5/E13 Physics-strengthened rule distillation**：A1 已完成调研和计划；A2/A3 已完成第一轮实验。Clip-level teacher signals + LOOCV model+rule selector 达到 0.6998，仅弱超 E12 0.6831。A4/A4b MLP pair scorer 多 seed 为负，mean train accuracy 0.9888 但 eval 0.5989 ± 0.0127。A5 window-level temporal selector 是首个显著正结果：no-leak clip-CV 0.9014，session-CV 0.8251，超过 G4/E11 dual fusion mean 0.7516。A6 GPU-trained neural MLP window selector 为负：session-CV 0.5898 ± 0.0068。
- **G5/E14 Shoulder-local vector spatiotemporal model**：A1/A1b/A2 已完成首轮。Best stable vector variant 是 image-basis + scale norm + smooth15，z-jitter 为 raw 的 0.4127x，但 corr 低于 raw。A2 Spatial→Temporal neural pair scorer train acc 0.9955、eval 0.3510，负结果；该表示可作为 auxiliary token，但不应作为 standalone pair scorer。
- **G5/E20 Non-leaky multi-seed baseline**：已完成 4 folds x 3 seeds。MoBind direct custom 为 0.4947 ± 0.2326；MoBind realistic transfer 为 0.4504 ± 0.1780；E17 raw-IMU realistic transfer 为 0.4685 ± 0.2287。严格 one-session-out 下旧 transfer 路线均未超过 direct custom。
- **G5/E21 Large-scale shoulder-local vector pretraining**：已完成 A0-A8。A1 source sanity 显示 hybrid raw-pose + vector 表示最强（0.7427 vs chance 0.3439）。A2/A3 true hybrid source pretrain -> custom one-session-out fine-tune 初始 3 seeds 为 0.6261 ± 0.1878；A7 扩展 seeds 1/2/3 后，6 seeds over 24 fold-runs 为 0.6550 ± 0.1823。A6 same-architecture direct custom/no source pretrain 为 0.5120 ± 0.1987；source-pretrain fine-tune 相比它提升明显。A8 full label-shuffle control 为 0.4628 ± 0.1429，aligned-shuffle paired delta +0.1923，bootstrap CI [+0.1123, +0.2769]。因此 hybrid + realistic source pretrain + custom fine-tune 是当前最强 learned transfer 组合，且不再只依赖 seed0 control。
- **G5/E22 IMU spectral/audio-style features**：已完成 A1 one-session-out diagnostic。Time-domain baseline weighted 0.6709，spectral-only 0.5316，time+spectral 0.5443。当前 spectral preprocessing 为负，不启动 heavy spectral neural branch。
