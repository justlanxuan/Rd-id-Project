# Plan：Source-to-Custom 骨架与跨模态域差异分解

## 0. 执行原则

1. G9 是独立科学目标，不能修改 G6 的 protocol、结果或 checkpoint。
2. 所有正式任务使用 G6 的 `window_len=24`、`stride=16`、四个 Custom outer sessions 和 seeds `0/42/123`，除非新建显式 protocol version。
3. 先做数据审计和单元级诊断，再启动训练；不以模型调参替代 gap 定位。
4. 所有大型 artifact 写入 `/data/fzliang/reid-project/g9`，仓库只保留配置、脚本、manifest 摘要和 Markdown 结果。
5. 2D、3D、SMPL 输入分别建轨；不同坐标空间不得直接比较。
6. 所有正式结果保留 raw `correct/total`、prediction details、config/data/protocol hash 和 provenance。

## 1. 目标架构

```text
source artifacts
  -> source manifest / content fingerprint
  -> per-window quality and motion features
  -> IMU/skeleton alignment diagnostics
  -> controlled source-to-custom matrix
  -> complexity/tracking/session stratification
  -> adaptation interventions
  -> validated gap report
```

G9 不新增一套训练 Dataset。复用 G6 canonical schema、WindowAlignmentDataset、ModelRegistry 和 MetricRegistry；新增内容只属于 gap feature、manifest、分层评测和诊断报告。

## 2. 实验分解

### E1：骨架源资产审计

目标：确认每个骨架源的真实内容、格式、表示空间和 provenance。

任务：

1. 构建 source manifest，记录 dataset、session、sequence、method、source type、joint order、space、normalization、confidence、missing、track 数和 content hash。
2. 校验 finite、shape、joint mapping、时间单调性和 canonical H36M-17 转换。
3. 对所有源计算 joint/bone/confidence/missing/tracklet 质量统计。
4. 对相同覆盖率或相同 sequence 的算法计算 pairwise hash、相关性和数值重复率。
5. 建立 source availability matrix，明确哪些算法在 source、Custom、验证 smoke 中真实存在。

完成门：无空/全零/非法 mapping；每个候选源有 provenance；疑似重复产物被标记并排除或解释。门禁按 source 独立输出 `included/conditional/excluded/pending`，不要求所有候选源同时通过；至少一个 source 和一个 Custom target 形成可重算的最小可信子集即可继续 E2/E3。

### E2：IMU 与跨模态配准审计

目标：区分 sensor marginal gap 与 IMU-skeleton relation gap。

任务：

1. 比较 raw IMU 和 normalized IMU 的 channel statistics、PSD、energy、jerk、autocorrelation。
2. 比较传感器位置、坐标系、单位、quaternion 顺序和采样率。
3. 计算 IMU 与 wrist/forearm skeleton velocity 的 cross-correlation 和最优 lag。
4. 计算 CCA/HSIC 或等价跨模态关系指标。
5. 执行 IMU-only、skeleton-only、fusion 的 source→Custom 对照。

完成门：输出 `imu_distribution.json`、`cross_modal_alignment.json` 和逐 session 表。

当前实现将这三类证据合并保存在 `/data/fzliang/reid-project/g9/e2_multimodal/multimodal_motion_diagnostics.json`，并明确 2D/3D representation、7D/legacy48 IMU layout 和采样/重采样 provenance；D5 已在四个 held-out Custom session 上完成 embedded 7D quaternion invalid-fill-only 与 unit-normalized 的固定检查点对照。

### E3：骨架 representation/source sweep

目标：测量不同 skeleton source 对 transfer 的影响。

第一层 source-side sweep：固定 Custom skeleton 为正式 AlphaPose/YOLO-Pose high，改变 source skeleton：

```text
TotalCapture GT
EgoHumans pose2d
AlphaPose
YOLO-Pose high
FMPose3D
MotionAGFormer
TCPFormer
WHAM
```

第二层 target-side sweep：固定 source skeleton 和 checkpoint，改变 Custom skeleton：

```text
Custom AlphaPose
Custom YOLO-Pose high
Custom 3D source（仅在真实 smoke 后追加）
```

第三层 matched-method sweep：仅对 source 和 Custom 都有真实输出的方法配对比较。

每个正式条件至少保存 source test、zero-shot、fine-tune/direct 的 raw counts；screening 可先单 seed，进入主表必须使用三 seed。

完成门：2D、3D、SMPL 分轨报告；不可把表示空间差异误报为算法优劣。

### E4：动作复杂度分析

目标：判断 Custom 性能下降是否集中在动作长尾。

任务：

1. 计算 motion energy、mean/max velocity、jerk、谱熵、periodicity、active-joint ratio、simultaneous-motion ratio 和 transition count。
2. 按低/中/高复杂度分桶。
3. 计算每个 source/session/condition 的 FrameAcc 曲线。
4. 做 complexity-matched evaluation，控制动作组成差异。

完成门：报告复杂度分布和每档 `correct/total`，不只报告整体平均。

当前 screening 输出每 source 的 low/mid/high motion-energy tertile 与运动特征；D6 已将 S06 528 个逐序列预测接入 pooled complexity、visibility 和 fragmentation-proxy 的 `correct/total` 分层。

### E5：时间、跟踪与身份关联分析

目标：区分 temporal protocol gap 和 representation gap。

任务：

1. 在 `lag ∈ {-8,-4,-2,0,+2,+4,+8}` 上扫描 IMU-video 对齐。
2. 固定 window 24/stride 16，补充 stride/window 诊断但不混入主协议。
3. 按 candidate group size、singleton、tracklet length、fragmentation、ID switch、confidence 和 occlusion 分层。
4. 同时保留 instantaneous Hungarian 和 tracklet-history 诊断结果。
5. G8 的 history 策略不作为默认改进，只用于定位状态/身份误差。

完成门：输出 lag、tracklet、candidate-group 与逐 session 结果；所有历史状态在 session 边界 reset。

当前 `/data/fzliang/reid-project/g9/e2_multimodal/tracking_quality.json` 已输出 S06 coverage、candidate group、tracklet fragmentation 和 baseline visibility delta；D6 已把 visibility/fragmentation proxy 接到预测分层，但由于输出没有独立 track IDs，ID switch 仍明确标记为不可识别。

### E6：适配干预

目标：验证可控修复是否对应已识别 gap。

按以下顺序执行：

1. source normalization；
2. target normalization；
3. pooled normalization；
4. frozen source encoder；
5. frozen encoder + linear adapter；
6. 只微调 IMU 分支；
7. 只微调 skeleton 分支；
8. 只微调 fusion/head；
9. full fine-tune；
10. CORAL/MMD/domain-adversarial（仅在前面证据支持时）。

所有干预必须使用相同 held-out protocol，禁止根据 test session 选择参数。

### E7：汇总与结论

目标：自动生成可审计的 gap report。

输出：

```text
gap_profile.json
g9_final_gap_manifest.json
skeleton_source_quality.json
imu_distribution.json
cross_modal_alignment.json
complexity_stratified_results.json
tracking_stratified_results.json
source_target_matrix.json
adaptation_results.json
results.md
```

每个结论必须标注：相关性证据、干预性证据、未解释因素和适用范围。

## 3. 正式结果矩阵

### Screening

- 所有正式骨架源：E1 审计，不训练；
- IMU-only/skeleton-only/fusion：每个代表性 source/session 先跑 seed 0；
- lag、复杂度、tracklet：先做单 fold/单 session smoke。

### Main matrix

- source：TotalCapture、EgoHumans；
- source skeleton：通过 E1 的正式源；
- target skeleton：Custom AlphaPose、YOLO-Pose high，以及真实 smoke 后的新增源；
- adaptation：zero-shot、fine-tune、direct；
- seed：0、42、123；
- Custom：四个 held-out sessions，沿用 G6 fold manifest；
- metrics：逐 session、macro-session、micro/weighted、`correct/total`、session/seed std。

正式矩阵在 E1/E2 门通过并生成 G9 protocol record 后才展开。

## 4. 测试与验收

- source manifest 能拒绝空、全零、非法 joint order、NaN、重复或缺 provenance artifact；
- 2D/3D/SMPL 表示空间不会被静默混合；
- IMU 和 skeleton frame/session/person 对齐可验证；
- lag、复杂度、tracklet 分层均能从 raw prediction 重算；
- Custom held-out session 不进入 normalization、checkpoint selection 或调参；
- 每个正式结果包含 protocol/config/data/checkpoint/content hash；
- 汇总器能检测 missing seed、重复 run、hash mismatch、失败 run；
- 至少一次真实非缓存 smoke 验证每个新增骨架后端；
- `python -m pytest -q`、Ruff、compileall、CLI help 和 `git diff --check` 通过。

## 5. 资源与调度

- artifact root：`/data/fzliang/reid-project/g9`；
- 先使用既有、通过 schema/content/provenance 校验的缓存；
- 新骨架后端按独立环境、权重和 GPU smoke 运行；
- 不在当前 `/home` 近满状态下写大型结果；
- 正式训练只使用显式 GPU 列表和可恢复 job manifest；
- 半写 checkpoint、run record 或预测禁止自动覆盖。

## 6. 交付物

- G9 formulation、survey、ideas、plan；
- E1 gap audit 的 manifest、测试和结果；
- 所有正式骨架源的质量与 provenance 表；
- IMU、cross-modal、复杂度、tracking、时间 gap 报告；
- source/target skeleton sweep 和 adaptation 结果；
- 自动生成的 `results.md` 与机器可读 JSON；
- `/data/fzliang/reid-project/g9/g9_final_gap_manifest.json` 及其输入 hash；
- HAROS progress/resume/experience 更新；
- 经人类审核后再决定是否创建 Git commit。
