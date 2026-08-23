# E4.1 Plan：增强朝向信号并验证转向配对

## 核心判断

AlphaPose 具有有限朝向信息：左右肩线、髋线、肩髋相对几何和其时间变化可形成 2D torso-heading proxy；但它不提供 3D body/root orientation，也无法从单目 2D 可靠区分“人物转身”和“相机/人物横向运动”。因此要提高 orientation 对模型的作用，必须分别增强信号、保证 source-target 同源、让训练目标关注转向，而不是只把 2D angle 拼到 embedding。

## 实验矩阵

### A. 表示 ablation

1. no orientation baseline；
2. AlphaPose 2D shoulder-line proxy；
3. 2D shoulder + hip axis + torso aspect/foreshortening + confidence/validity；
4. camera-motion-compensated 2D proxy（使用 bbox/global optical-flow control）；
5. 2D→3D lifting（FMPose3D、MotionAGFormer、TCPFormer 分开）后派生 3D torso heading；
6. WHAM canonical/world heading（仅在 target/source 坐标系可审计时进入主表）。

### B. 模型归因

1. orientation-only ↔ skeleton-only ↔ gyro-only；
2. concat、learned gate、cross-attention、turning mixture-of-experts；
3. orientation→skeleton fusion 与 orientation↔gyro cross-attention 分开；
4. turn-conditioned gate：由 `turning_activity` 控制 orientation branch 权重，低转向时接近 baseline。

### C. 训练目标

1. 原始 InfoNCE；
2. orientation-rate ↔ gyro-magnitude 辅助对比损失（不强行对齐未知 gyro axis）；
3. 高转向窗口加权 InfoNCE / pairwise ranking；
4. turn-onset temporal consistency；
5. hard negatives：同 session、相似 pose 但不同人物/不同转向阶段；
6. orientation pretrain 后再做 Re-ID finetune。

## 数据与评估约束

- 主目标只为 Custom23（全局转向）；57/22/24 是 non-turning negative controls，只检查不恶化。
- source train 与 Custom23 必须使用同一 extractor、同一 17-joint contract、坐标归一化和 visibility 语义；当前 `skeleton_source=gt` 训练路线不得直接与 Custom AlphaPose 方向结果混称。
- 每个 session 按 low-turn / high-turn 分层；主指标为 Custom23 high-turn FrameAcc、raw correct/total 和 margin，secondary 为全 23、低转向 23 以及 57/22/24 negative controls。
- 至少 3 seed；所有表示/模型/损失组合先做小矩阵 screen，再对最优 2–3 个做长训练和 hard-negative 复验。

## 晋级标准

orientation 模型必须在 Custom23 high-turn 子集稳定优于 no-orientation baseline，并在 low-turn 23 与非转向负控不产生系统性退化；不能以全 session 平均或单 seed peak 代替转向分层证据。

## 已执行的 screen 与后续

- 已完成同源 TC-only 的 2D proxy、3D heading、turning-gate、gyro-cross 和可学习
  gyro-activity auxiliary screen（每项 3 seeds）。
- 初步最优为 3D heading + gyro cross + auxiliary：high-turn 有收益，但 low-turn
  明显下降，属于“转向专门化”证据而非整体晋级。
- 下一轮固定训练/选择 epoch，增加 5–10 seeds；然后分别加入 high-turn weighted
  InfoNCE、turn-onset consistency 和 session 内 hard negatives。最终只在冻结的
  Custom23 high/low 划分上报告一次确认结果，并保留 57/22/24 负控。

## Frozen confirmation extension preregistration

第一轮 frozen confirmation 后，100% same-action hard negatives 出现近 chance 的
训练塌缩。后续扩展只使用 `custom23_validation` 调参，测试以下预先固定组合：

1. conditional-cross + auxiliary，turn weight=`0/2/4`；
2. conditional-cross + auxiliary + weight=`2`，hard fraction=`0.25/0.50`；
3. 每项 3 seeds、5 epochs×75 steps，仍按 validation high-turn 选 checkpoint。

conditional-cross 使用观测到的 turning activity 作为 residual gate：低转向时退回
原 skeleton/IMU baseline，高转向时才注入 orientation/gyro residual。扩展晋级门槛
在运行前冻结为 validation high-turn 均值至少 `0.573`（比 frozen baseline `0.540`
多一个正确查询/30），validation low-turn 不低于 `0.458`，且 3 seeds 不出现训练
塌缩。未过门槛不得再次查看 frozen test；过门槛后只对晋级配置运行一次 5-seed
confirmation。

Turn-onset extension 固定在 validation-only，使用已注册的 conditional-cross、
aux=`0.05`、turn weight=`2`，仅比较 onset weight=`0.05/0.10/0.20`，每项 3 seeds。
两侧分别从本侧 embedding 预测同一个 8-bin onset target；禁止任一 inference tower
读取另一模态。晋级门槛继续使用 high-turn≥`0.573`、low-turn≥`0.458`、无塌缩，
不因本轮结果修改。

## Physical turning-MoE confirmation preregistration

Validation-only physical screen 冻结出如下 comparator，不再调参：

- skeleton expert：MotionBERT 3D torso heading 的 `abs(rate)`；
- IMU expert：逐帧 gyro magnitude；
- score：Pearson correlation 在 lag=`[-2,-1,0,1,2]` 中取最大值；
- routing：group turning count ≥ `19/48`（等价 validation-only Q75）
  时使用 physical score，否则回退该 seed 的 frozen baseline dot-product；
- 不学习 test calibration，不混合两个 score 的权重。

该 MoE 在 validation 为 full `57/98=0.582`、high `14/24=0.583`、low
`43/74=0.581`。下一步只运行一次 Custom23 frozen test 和 57/22/24 controls；
报告 raw correct/total，并同时报告 physical-only 与 baseline，禁止继续按 test 调整 lag、
threshold 或 routing。

Negative-control audit 后另注册一个明确标注为 post-hoc safety 的 persistence router：
count≥19 且 ±24 source frames 内至少还有一个 high group 才启用 physical expert。
原因是全局转向应跨越相邻重叠窗口，而孤立 spike 更可能是 3D lifting 噪声。该规则
用于判断能否消除 Custom24 的两个孤立误触发；由于它在查看 control 后提出，不得
作为首次 confirmatory 结果，需后续独立 session 才能正式晋级。
