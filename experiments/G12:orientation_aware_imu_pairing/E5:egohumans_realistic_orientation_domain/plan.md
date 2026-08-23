# E5 Plan：EgoHumans realistic 朝向与跨 source 训练

## 1. 研究问题

E4.1 在 Custom23 上证明了 3D-derived torso heading 可作为转向窗口的物理配对信号，但训练主线主要使用 TotalCapture。E5 检验：

1. EgoHumans realistic source 内，朝向/朝向变化是否能改善 skeleton–IMU 配对；
2. EgoHumans-only、TotalCapture-only 和两 source 共同训练谁更稳健；
3. 朝向提升来自可迁移的 heading-rate/activity，还是来自 source-specific 坐标/采集偏差。

E5 不修改 G10/G11 freeze，也不把旧 346-row Custom 结果升级为新结论。

## 2. HAROS 数据门禁与表示分轨

### 2.1 时间窗口

统一使用 0.8 秒、输出 24 个时间点：

| source/target | native rate | native frames | model points |
|---|---:|---:|---:|
| EgoHumans realistic | 20 Hz | 16 | 24 |
| TotalCapture | 60 Hz | 48 | 24 |
| Custom | 30 Hz | 24 | 24 |

旧 G10 Ego `windows_train.csv` 的 24 帧在 20 Hz 下是 1.2 秒，只保留为 superseded control。

### 2.2 Orientation tracks

| track | source | orientation | 用途 |
|---|---|---|---|
| O0 | all | no orientation | baseline |
| O2D | Ego S06 AlphaPose / Custom MotionBERT-AlphaPose | 2D shoulder-line π-periodic proxy | same-2D-proxy control |
| O3D | Ego/TC canonical 3D joints、Custom MotionBERT 3D joints | 3D torso heading-derived rate | 主 orientation track |
| O3D-rate | O3D | rate/activity only，屏蔽绝对 sin/cos | 坐标系泄漏控制 |

EgoHumans raw SMPL `global_orient` 仅作 direct-orientation reference；当前 canonical pair input 未保留该字段，不直接把它当作模型输入。

O2D 与 O3D 分表。O3D 是 joint-derived，不称为 extractor direct orientation；O2D 的坐标系是 image plane，不能称为 world yaw。

## 3. 训练矩阵

统一 skeleton/IMU 输入、hidden=96、embedding=64、batch=32、AdamW lr=2e-3、temperature=0.1、0.8 秒窗口和同一 InfoNCE 训练预算。

### 3.1 Source regime

| regime | train data | 目的 |
|---|---|---|
| EH-only | EgoHumans realistic train sessions | source 内朝向验证 |
| TC-only | TotalCapture train | E4.1 对照 |
| TC+EH-balanced | 两 source group-balanced sampler | 共同训练主候选 |
| TC+EH-unbalanced | 两 source 原始比例 | 样本量/比例控制 |

Ego validation 按 session 划分，不按重叠窗口随机划分；Custom23 validation/test 使用已冻结 group protocol。

### 3.2 Model variants

每个 regime 至少运行 O0、O2D、O3D、O3D-rate；screen 采用 3 seeds，晋级候选采用 5 seeds。保留 `turning_cross + gyro-activity auxiliary` 作为 learned orientation candidate；physical turning-MoE 作为冻结 baseline checkpoint 上的 deterministic comparator，不与 learned fusion 混称。

## 4. Evaluation

1. EgoHumans held-out sessions：报告 group-level `correct/total`、FrameAcc、margin、orientation coverage；主选择只看 Ego validation，不查看 Ego test。
2. Custom23：使用 `/data/fzliang/reid-project/g12/e4_1_source_aligned/motionbert_alphapose_cache_v2/manifests/custom23_frozen_protocol.json`，报告 high-turn、low-turn、full。
3. Custom57/22/24：只作 non-turning negative controls，检查是否退化。
4. 每个 checkpoint 的 test 只评估一次；保存 config、manifest、checkpoint、raw metrics 与 SHA256。

## 5. 晋级标准

候选训练方式必须同时满足：

- EgoHumans held-out 相对同 regime O0 有稳定 orientation gain；
- Custom23 high-turn 相对同 regime O0 有稳定 gain；
- Custom23 low-turn 与 57/22/24 不出现系统性退化；
- 3/5 seeds 方向一致，不能只报告单 seed peak；
- O3D-rate 若不弱于 O3D，则优先选择 rate/activity 版本，降低坐标系依赖。

## 6. 当前执行顺序

1. 已生成 Ego session-level train/validation manifests，并验证 16-frame native window；
2. 已完成 source-aligned O0/O2D/O3D/O3D-rate screen 的可执行子矩阵（O3D full/rate、O2D proxy；rate 作为屏蔽绝对 heading 的控制）；
3. 选择门禁固定在 validation，Custom23 high 用于目标侧晋级，Ego AlphaPose validation 用于 source-side safety audit；
4. 已对 TC-only O0 与 O3D `turning_cross` 做 5-seed confirmation；
5. 结果显示 fully-aligned TC-only O3D cross 是当前 Custom23 的相对最佳配置，但 full 仅约 +2.0 pp、Ego validation 无提升，不能宣称稳定朝向收益；EH-only、TC+EH balanced 与 O2D proxy 均不晋级。七个 Ego canonical test session 缺少 source-aligned cache，后续若补齐再做独立 test confirmation。
