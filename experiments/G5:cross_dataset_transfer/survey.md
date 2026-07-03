# G5 Survey: Cross-Dataset Transfer 现有基线与经验教训

## 1. 源数据集：EgoHumans

- **数据规模：** 公开 egocentric 多人视频数据集，包含 128 个序列，官方 action split 为 98/6/24（train/val/test）。
- **IMU 设置：** MoBInd 官方使用 5 个肢体（LeftWrist, RightWrist, LeftKnee, RightKnee, Head）的 synthetic IMU。
- **G_egohumans 控制变量结论（cache bug 修复后）：**
  | 实验 | IMU | 窗口 | FrameAcc | 关键结论 |
  |---|---|---|---|---|
  | E6-correct | 1 IMU / RightWrist | 24 帧 | **0.9548** | 同关节 IMU↔pose 匹配极强 |
  | E8 | 1 IMU / RightWrist | 100 帧 | **0.9616** | 长窗口收益小 |
  | E9 | 5 IMU | 24 帧 | **0.9641** | 多 IMU 略好 |
  | E7 | 5 IMU | 100 帧 | **0.9675** | 全设置上限 |
- **限制：** E6/E8 的 motion 侧被限制为与 IMU 同名的单个肢体，做的是“同关节信号匹配”，不是真正的“单 IMU + 全视频骨架”匹配。

## 2. 目标数据集：Custom

- **数据规模：** 4 个 session（20260211_171423, 171724, 172257, 172522），每个 session 约 30–60 分钟，2 人佩戴 IMU。
- **切分方式：** per-video 7:3 split，每段按 `segment_frames=1800` 切分，前 70% train、后 30% test，val 取自 train 前 10%。
- **IMU 设置：** 实验中主要使用 `RightWrist`（与 EgoHumans 单 IMU 设置对齐）。
- **当前 SOTA（G4/E11）：** 单 IMU RightWrist + dual-embedding（local + global）fusion，w24 FrameAcc = **0.752 ± 0.095**（6 seeds）。

## 3. 历史跨数据集迁移尝试

### G_egohumans/E9: E8 → custom 迁移（旧 Autism pipeline）
- **方法：** 加载 E8（EgoHumans 单 IMU 100 帧）检查点，在 custom 4-fold 上做 zero-shot / frozen adapter / full finetune。
- **结果：**
  - zero-shot: 0.339
  - frozen adapter: 0.339（无效）
  - from-scratch: 0.578 ± 0.100
  - finetune: 0.557 ± 0.103
- **结论：** 直接迁移困难；frozen adapter 完全无效；from-scratch 与 full finetune 相当。

### G_egohumans/E10: EgoHumans + custom 联合训练（旧 Autism pipeline）
- **方法：** 在 EgoHumans train + custom per-video 7:3 上联合训练。
- **结果：**
  - EgoHumans test: 0.719 ± 0.011
  - custom test: 0.611 ± 0.078
- **结论：** custom 优于 E9，但方差大；与 E10b（custom-only same split，0.613 ± 0.010）无显著差异，说明提升主要来自切分策略而非 EgoHumans 数据。

### G4/E11: custom 上 dual-embedding from-scratch（MoBInd）
- **方法：** 分别训练 Model-L（RightWrist ↔ RightWrist motion）和 Model-G（RightWrist ↔ full pose2d），推理时 score-level fusion。
- **结果：**
  - w24 Fusion best α: **0.752 ± 0.095**
  - w100 Fusion best α: 0.723 ± 0.124
- **结论：** dual-embedding 在 custom 上有效，但 Local 与 Global 错误高度正相关，互补性低于理想独立专家。

## 4. 可借鉴方向

1. **源域预训练 + 目标域 fine-tune：** 在 EgoHumans 上预训练 dual-embedding，再在 custom 上 fine-tune，可能同时获得源域泛化性和目标域适配性。
2. **分离 Local / Global 迁移策略：** Local branch 学习的是同关节运动，可能在跨数据集时更稳定；Global branch 学习的是全身姿态，可能受 camera view / 人数影响更大。
3. **IMU 跨域归一化：** EgoHumans synthetic IMU 与 custom real IMU 的统计分布差异大，可能需要 statistics alignment 或 domain-invariant losses。
4. **渐进式 fine-tune：** 先冻结 encoder 训练 adapter，再逐步解冻，避免破坏源域学到的对齐表示。
5. **多源数据联合：** 不仅 EgoHumans，未来可引入 TotalCapture 等实验室数据集，构建更通用的单 IMU 预训练表示。

## 5. 当前空白

- 没有使用 **MoBInd dual-embedding 架构** 做源域预训练 + 目标域迁移的实验。
- 没有系统比较 zero-shot / fine-tune / adapter / progressive unfreezing 在 MoBInd 架构下的效果。
- 没有分析 Local branch 与 Global branch 各自的跨域迁移能力。
