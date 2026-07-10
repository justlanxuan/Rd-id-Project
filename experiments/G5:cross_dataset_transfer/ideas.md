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

- **描述：** 分别验证 Local branch（LeftWrist ↔ LeftWrist motion）和 Global branch（LeftWrist IMU ↔ full pose2d）的迁移能力。可能 Local branch 因任务更简单而迁移更稳定。
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

## I9: Realistic single-source 保守 fine-tune

- **描述：** E4b 显示 realistic single-source 是目前最强迁移源，但 full fine-tune 后均值仍低于 direct custom dual fusion。尝试冻结 encoder、只训练 projection/head、progressive unfreezing 和低学习率 full fine-tune，避免 custom 小数据破坏 source representation。
- **验证：** 已验证有效。E3-A4 low-LR full fine-tune 6 seeds = `0.7413 ± 0.0241`；在有 matched baseline 的 seed0/42/123 上均超过 E4b `lr=1e-4` full fine-tune。
- **结论：** 在发现 custom 实际为 LeftWrist IMU 前，这是最佳 transfer 方法；现在仅作为 historical mislabeled-wrist baseline。后续应以 E9 LeftWrist 复验结果为正式 baseline，而不是继续扩大 MoBind-like source。

## I9b: Realistic dual-embedding 保守 fine-tune

- **描述：** 将 I9 的低学习率策略迁移到 G5/E2 local/global dual branches，检查是否能修复 seed123 collapse。
- **验证：** 已做 E4 seed123 诊断。`none=0.4861`、`zscore=0.4894`、`minmax=0.4894`，相比 E2 seed123 约 +0.066 到 +0.081。
- **结论：** 部分有效但不值得扩展：仍远低于 E3 seed123 `0.7603`，且训练成本为两个 branch。

## I10: Single-wrist adaptive local/global gate

- **描述：** 左手腕 IMU 的局部可靠性随动作片段变化。手腕运动丰富时 local branch 更可靠；手腕静止/遮挡时 global skeleton branch 可能更可靠。用 IMU energy、LeftWrist 2D velocity、visibility、score entropy/margin 来预测 fusion α。
- **验证：** E5-A1 oracle diagnostics 已完成。G4 direct w24 frame oracle 有上限空间（`0.8494 ± 0.1009`），但 G5/E2 fine-tune minmax clip oracle 只有 `0.6877 ± 0.1417`，低于 E3 low-LR transfer。
- **结论：** 对 direct custom dual/deployment 稳定性值得继续；对当前 E2 transfer dual，alpha/gate alone 不足以超过 E3。

## I11: IMU canonicalization + domain randomization

- **描述：** TotalCapture、EgoHumans、realistic synthetic、custom 的 IMU 坐标系、重力、单位、sensor order 不完全一致。建立 metadata 并统一为 canonical representation，同时训练期随机注入旋转、scale、bias、noise、time shift 和 sensor dropout。
- **验证：** E6-A1 IMU format audit 已完成。E4b/E3 的 20 Hz custom cache 文件名为 RightWrist，但 custom 实际佩戴在 LeftWrist；E9 已重建 explicit LeftWrist source/custom cache。TotalCapture 48D sensor order 是 low-leg/low-arm，不直接匹配 left-wrist-only deployment。
- **验证补充：** E6-A2/A3 target train-only light acc jitter 已完成 seeds 0/42/123，结果与 E3 low-LR 持平，mean delta `-0.0000`。
- **结论：** target-only light jitter 无收益。下一步若继续 canonicalization，应做 source+target 同步变换，或先明确 custom acc 的 gravity/frame 语义；暂缓 TotalCapture multi-source。
- **风险：** custom acc 的 gravity/frame 语义仍未完全记录；过强扰动会损害身份判别信号，尤其在单左腕 IMU 已经信息稀疏时。

## I14: LeftWrist-correct realistic pretrain and direct custom

- **描述：** custom IMU 实际佩戴在 LeftWrist，因此需要重新验证最高性能路线：realistic LeftWrist source pretrain -> custom LeftWrist zero-shot / low-LR fine-tune，并补 direct custom LeftWrist from-scratch 对照。
- **验证：** E9 已完成 matched seeds 0/42/123。Realistic LeftWrist source -> custom zero-shot 为 `0.6115 ± 0.1409`，low-LR fine-tune 为 `0.6148 ± 0.1638`；direct custom LeftWrist from-scratch 为 `0.6346 ± 0.1681`。
- **结论：** corrected LeftWrist transfer 没有超过 direct custom，也明显低于历史 mislabeled E3 `0.7413 ± 0.0241`。不扩展到 6 seeds；下一步应诊断 seed123 collapse / motion-side sensitivity，而不是继续同一 transfer recipe。

## I15: Skeleton differential / relative motion representation

- **描述：** 当前 MoBind motion side 主要使用 normalized 2D keypoint positions。对于 wrist IMU，速度、加速度、骨段角度、角速度、相对 body center motion 可能比绝对 keypoint 更接近 IMU 物理量。
- **验证：** E10-A1 已完成 cache-level diagnostic。`smooth_kernel=5` 时 top pairs 为 forearm length/angle 与 IMU acceleration/change；raw wrist position 仍有竞争力。Filter sweep 显示平滑能提高 top lag-tolerant correlation：kernel1 `0.3613`，kernel5 `0.3956`，kernel9 `0.4341`。
- **补充验证：** A1b 显式加入大臂-小臂夹角及其变化速度。smooth9 下 `elbow_included_angle` vs `imu_acc_norm` 的 corr 为 `0.4276`；elbow angular velocity 的 window energy corr 最高到 `0.7050`。
- **结论：** 不应直接用“纯差分”替代 keypoints；更合理的是 hybrid 表示：保留低频骨架几何（forearm length、elbow angle、relative position），加入 filtered elbow angular velocity、wrist velocity 和 body-center velocity。1D temporal CNN 作为低通前端的假设得到初步支持。
- **后续更新：** E11 已按这个方向构建更完整的 50D shoulder-anchored kinematic cache。A1 traditional matcher 有效，但 A3 MoBind feature-path 失败，因此不要继续普通 motion-feature cache replacement；改走 I16 的专用 matcher / auxiliary branch。

## I16: Shoulder-anchored kinematic matcher / auxiliary branch

- **描述：** 不再把骨架压缩成少量 scalar，也不单纯替代 keypoint。以 LeftShoulder 为局部支点，显式描述 upper arm、forearm、elbow included angle、wrist relative motion，并注入 LeftShoulder / BodyCenter global context。目标是从相对位置关系和骨架活动情态描述 motion，使其更接近 LeftWrist IMU。
- **验证：** E11-A1 no-training traditional matcher 已完成，FrameAcc `0.6352`，超过 E10-A3 learned `hybrid_v1=0.5330`，说明 kinematic signal 可以直接转化为 matching。E11-A2/A3 又完成 50D `shoulder_kinematic_v1` cache + MoBind feature-path seed0，结果仅 `0.3355`。
- **验证补充：** E11-A4/A5/A6 已完成。Learned lag-corr linear scorer 为 `0.6136`；clip-global temporal assignment 为 `0.6831`；simple agreement policy 为 `0.6881`；oracle window/global selector 为 `0.8914`。
- **验证补充 2：** E11-A7/A8/A9/A10 已完成。A7 train-supervised orientation 与 A8 Viterbi 为负；A9 signed lag-correlation 将 window matching 提到 `0.6586`；A10 hybrid signed-window / abs-global consistency selector 达到 `0.9014`，超过 raw-keypoint direct seed0 `0.8396`。
- **验证补充 3：** E11-A11 已完成 frozen A10 rule 的 multi-seed/bootstrap validation。A10 `0.9014` 超过 direct custom seeds 0/42/123，并且相对 3-seed per-clip mean 的 bootstrap CI 为正。
- **验证补充 4：** E12 已完成 realistic source-trained selector 多 source-seed 验证。6 folds 全部选择 `abs_global`，custom `0.6831 ± 0.0000`，超过 MoBind-like/E9 corrected means，但低于 custom-selected A10 `0.9014`。
- **结论：** 想法方向成立，且 source-trained kinematic matching 已超过旧 MoBind-like transfer。但现有 MoBind `motion_type: feature` 接口不适合这类异质 kinematic channels；A10 的大幅提升还没有被 source-trained selector 复现。下一步若继续，应做无需 custom test-sweep 的 selector/专用 lag-correlation 或 cross-attention matcher。
- **风险：** custom val 只有 2 windows，传统 MoBind early stopping 对 kinematic channels 很不可靠；需要更稳的 validation / distillation / temporal matching objective。

## I17: Physics-strengthened rule distillation

- **描述：** 不把 E11-A10 rule 当最终部署方案，而是把 signed-window、abs-global、A10 selector、score margin、assignment consistency 蒸馏到可训练模型。模型保留 raw pose，同时加入 shoulder-anchored kinematic auxiliary branch、lag/correlation objective 和 assignment-aware loss。
- **调研：** E13-A1 已整理 rule distillation、Soft-DTW/differentiable temporal alignment、Sinkhorn/differentiable assignment、differentiable ranking 等可用思路与仓库。
- **计划：** E13-A2 先生成 teacher signals；E13-A3 实现 raw-pose + kinematic auxiliary student；E13-A4 加 distillation / lag / assignment losses；E13-A5 做 ablation。
- **当前实验：** E13-A2/A3 已启动并完成第一轮。A2 生成 7 个 clip-level teacher-signal 文件；A3 LOOCV 小 selector 达到 `0.6998`，仅弱超 E12 `0.6831`，不足以说明 pipeline 显著上升。
- **当前实验 2：** E13-A4/A4b MLP lag-corr pair scorer 多 seed 达到 mean train accuracy `0.9888`，但 eval 只有 `0.5989 ± 0.0127`，说明 pair-only student 过拟合且不能解决 temporal identity assignment。
- **当前实验 3：** E13-A5 window-level temporal selector 已完成。移除 GT-derived prototype feature 后，no-leak clip-CV 达到 `0.9014`，session-CV 达到 `0.8251`，首次在 held-out protocol 下超过 G4/E11 dual fusion mean `0.7516`。
- **当前实验 4：** E13-A6 GPU-trained MLP window selector 已完成多 seed。clip-CV `0.5908 ± 0.0450`，session-CV `0.5898 ± 0.0068`，说明当前小数据下 neural selector 不如 ridge，不能替代 A5。
- **当前实验 5：** E14 shoulder-local arm-vector Spatial→Temporal idea 已完成首轮验证。稳定 vector 表征可降噪，但 standalone neural pair scorer seed0 只有 `0.3510`，train pair acc `0.9955`，说明问题仍在 pair objective/temporal assignment，而不是仅换骨架表示。
- **成功标准：** 先超过 E12 deployable `0.6831`，强目标超过 G4/E11 dual fusion `0.7516 ± 0.0946`，最终逼近 E11-A10 `0.9014` 且不依赖 test-sweep 阈值。
- **下一步：** 冻结 A5 no-leak feature set，诊断 session-CV 失败 clip `custom_20260211_171423_seg1`，并把 window-level selector 扩展成更稳的 temporal/assignment-aware student。仅靠 7 个 clip-level confidence 特征或 pair-only scorer 都不够。
- **风险：** A10 teacher 本身有 split-specific bias；必须使用 confidence weighting、source+custom augmentation 和 held-out evaluation，避免把手写阈值过拟合直接蒸馏进模型。

## I12: MoBInd embedding 接入 Autism pipeline temporal logic

- **描述：** MoBInd 表示学习强，Autism pipeline 的 temporal smoothing、sliding vote、HOTA/AssA 评估更接近部署。将 MoBInd window-level similarity 作为 pipeline 后处理输入，利用时间一致性降低 frame-level 抖动。
- **验证：** 导出 MoBInd similarity matrix，接入 pipeline vote / identity consistency，比较 raw matching vs temporal post-processing。
- **风险：** 后处理可能掩盖 representation 问题，需要同时报告 raw FrameAcc 和 temporal metrics。

## I13: Dataset-specific normalization / adapter

- **描述：** 与其强行 DANN 做完全 domain-invariant，不如保留 shared encoder，同时给 TotalCapture / EgoHumans / realistic / custom 配 dataset-specific BN 或 adapter。
- **验证：** multi-source training 中比较 shared-only、DSBN、adapter、DANN。优先 DSBN/adapter，DANN 仅作辅助 baseline。
- **风险：** custom 数据量小，adapter 参数过多会过拟合。

## I18: Topology-aware arm-vector Spatial->Temporal model

- **描述：** E14 的 shoulder-local vector idea 不应继续用 token mean pooling。对于 LeftWrist IMU，关键关系是 upper/forearm/wrist/context 的有序拓扑、elbow angle、forearm relative rotation、closure consistency 和全局 body context。新模型应把左肩作为根节点，把左臂表示为 typed arm graph tokens，而不是无序 token bag。
- **设计：** E17 已建 HAROS。Skeleton 默认采用混合层次 token：bone-level `upper_bone/forearm_bone`，part-level `arm_part`，global-level `global_context`。每个 bone token 不只含 `(dx,dy)`，还包括方向、长度、平滑速度、加速度、相对 parent rotation 和可选 visibility。Part token 显式包含 `dot/cross`、elbow angle `(cos,sin)`、length ratio、closure residual 等 relation features。IMU 输入也不只用 flat 7D，而是构建 `acc / acc_dyn / ori / rot_dyn` typed physical tokens，同时提供 raw vector 与 invariant dynamics（acc magnitude、acc delta magnitude、quat angular speed）。每帧先用 topology-aware spatial attention / physical-token encoder 编码，再用 temporal CNN + GRU/Transformer 编码时间。
- **训练：** 不沿用 E14 binary pair classifier 作为主目标。优先使用 contrastive retrieval / InfoNCE、in-batch negatives 和 one-session-out protocol；E14-style mean pooling 与 raw 7D IMU tower 必须作为 ablation/control。
- **成功标准：** 至少超过 E14-A3 session-heldout `0.5026 ± 0.1250`；更强目标是超过 E16 MoBind one-session-out baseline 和接近 E13-A5 session-CV `0.8251`。
- **风险：** 数据量小，Transformer 可能过拟合；是否使用 Transformer 由 A4 temporal ablation 决定，而不是默认认为更大模型更好。

## I19: Large-scale shoulder-local vector pretraining

- **描述：** E14/E17 的 custom-only neural vector models 是负结果，但 E11/E13 证明 shoulder-anchored kinematic signal 本身有效。新的问题不是继续调 E14/E17，而是判断 shoulder-local vector / hybrid vector 表示是否需要在 EgoHumans realistic 或 MoBInd-like source 上先学到通用运动先验。
- **验证：** E21-A0/A1 已完成。Source-domain sanity 中 `hybrid` + time-domain IMU target 达到 `0.7427`，高于 `vector` `0.6967` 和 `raw_pose` `0.6590`，chance `0.3439`。
- **验证补充：** E21-A2/A3 完成 true raw-pose + shoulder-vector hybrid source pretrain -> custom session-out transfer。4 folds x 3 seeds = `0.6261 ± 0.1878`，超过 E20 MoBind direct custom `0.4947 ± 0.2326`，paired delta `+0.1313` with bootstrap CI `[+0.0135, +0.2451]`。
- **归因消融：** E21-A6 完成 same-architecture direct custom/no source pretrain。4 folds x 3 seeds = `0.5120 ± 0.1987`。这只比 E20 MoBind direct custom 高 `+0.0173`，而 E21 source-pretrain fine-tune 比同架构 direct custom 高 `+0.1141`，bootstrap CI `[-0.0027, +0.2524]`。
- **稳健性扩展：** E21-A7 新增 seeds `1/2/3`。新 12 fold-runs 为 `0.6840 ± 0.1718`；合并 seeds `0/1/2/3/42/123` 后为 `0.6550 ± 0.1823` over 24 fold-runs。结果完整性和 split/config 审计未发现直接 bug。
- **反捷径控制：** E21-A8 完成全量 label-shuffle control。Label-shuffle 为 `0.4628 ± 0.1429`，aligned 为 `0.6550 ± 0.1823`，paired delta `+0.1923`，bootstrap CI `[+0.1123, +0.2769]`。这比 seed0-only control 更有说服力。
- **控制：** E21-A5 source-only zero-shot 为 `0.4131 ± 0.1618`，低于 fine-tune；A4 seed0 label-shuffle fine-tune 仍有 `0.5631 ± 0.0879`，说明 source pretraining 自身有信号，但 anti-shortcut controls 应扩展到所有 seeds。
- **结论：** E21 是当前首个 learned source-transfer 正结果，且 6-seed 扩展和 full label-shuffle control 后仍成立；提升不能只归因于 hybrid 表达本身，最合理解释是 hybrid skeleton 表达、realistic source pretraining、custom fine-tune 三者叠加。第一版不要加入 spectral target，因为 `hybrid + time+spectral` 在 A1 中降到 `0.6987`。
- **优先设计：** skeleton 用 LeftShoulder-rooted arm graph、bone vectors、elbow relation、global context；IMU 默认保守使用 raw 7D + light dynamics，避免 E17 里过拟合严重的 typed physical token tower。
- **成功标准：** source-domain 明显高于 chance；custom one-session-out 下超过 E20 direct custom MoBind `0.4947 ± 0.2326` 和 E20 E17 realistic transfer `0.4685 ± 0.2287`。当前已达到，但 final claim 还应补全 label-shuffle controls 到多 seed。
- **风险：** 若 source 上学得好但 custom transfer 差，说明主要是 domain gap；若 source 上也差，说明 vector 表示/架构仍有根本问题。

## I20: IMU spectral / audio-style preprocessing

- **描述：** 把 IMU 视作一维时序信号，提取类似音频处理的多尺度频域和时频特征，例如 band power、dominant frequency、spectral centroid、spectral entropy、STFT/wavelet map。核心不是用频谱替代 raw IMU，而是作为 raw 7D 的辅助通道或 selector feature。
- **验证：** E22-A1 已完成 one-session-out low-capacity diagnostic。Weighted: time-domain `0.6709`，spectral-only `0.5316`，time+spectral `0.5443`。
- **结论：** 当前 spectral/audio-style preprocessing 不应进入 heavy neural training；若继续，只能作为重新设计的辅助分支或 learnable filterbank，并且必须先超过 time-domain diagnostic。
- **候选频段：** 20 Hz IMU 下 Nyquist 为 10 Hz；先用 `0.2-1 Hz`、`1-3 Hz`、`3-6 Hz`、`6-10 Hz`。同时给 skeleton 侧构造 elbow angular velocity、wrist relative velocity、body-center velocity 的对应频域/能量特征。
- **成功标准：** raw 7D + spectral auxiliary 在同一 one-session-out protocol 下超过 raw 7D only，并超过 E20 direct custom MoBind `0.4947 ± 0.2326` 或 E20 E17 transfer `0.4685 ± 0.2287`。
- **风险：** 纯 FFT magnitude 会丢相位和瞬态；如果只在 leaky/original split 有提升，应视为 session-specific shortcut。
