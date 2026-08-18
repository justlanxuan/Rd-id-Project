# E3 Results：Source/Target 矩阵索引

状态：`indexed_plus_s06_fixed_checkpoint_and_imu_controls`。已有 G6 结果仍按原协议索引；另外完成了 S06 六种骨架源的固定检查点坐标扫掠、Custom target IMU quaternion 对照和 prediction-level 分层。

产物：`/data/fzliang/reid-project/g9/e3_source_target/source_target_matrix.json`。

- G6 已有 66 个 completed evaluation records，聚合为 22 个 source/condition/session cells，保留每个 seed 的 `correct/total`、protocol hash 和 run-record 路径。
- 已有 source performance 只覆盖 canonical TotalCapture 和 EgoHumans；AlphaPose、FMPose3D、MotionAGFormer、TCPFormer、WHAM 目前只有 E1/E2 语义/运动证据，没有 G6 性能 cell。
- YOLO-Pose high 明确为 conditional，未进入任何正式性能排名。
- 四个 Custom session 的 target motion/IMU 特征已附到每个 G6 cell；zero-shot 的 target motion-energy 与 FrameAcc 相关性仅作描述性探索（每个 source 只有 n=4），不作因果结论。

### S06 fixed-checkpoint sweep

D3/D4 对 88 个 Custom 序列、6 种 S06 源、raw 与 `screen_calibrated` 两个坐标变体执行相同的 24/16 segment protocol，固定 baseline IMU、GT person order 和 EgoHumans source checkpoint。该实验是固定检查点的坐标干预，不是重训后的 source-domain benchmark；G6 encoder 只消费 xy，因此 3D 源在此被投影为 xy。

| source | raw FrameAcc | screen-calibrated FrameAcc | delta (screen−raw) |
|---|---:|---:|---:|
| AlphaPose | 0.29246 | 0.25700 | −0.03546 |
| YOLO-Pose high (conditional) | 0.26724 | 0.27164 | +0.00441 |
| FMPose3D | 0.23860 | 0.22581 | −0.01279 |
| MotionAGFormer | 0.26211 | 0.23927 | −0.02284 |
| TCPFormer | 0.26332 | 0.23489 | −0.02842 |
| WHAM | 0.24156 | 0.22992 | −0.01164 |

所有 12 个 cell 均完成，逐序列配对数为 528，无缺失。归一化干预没有普遍增益：仅 YOLO-Pose high 略升，其余五源下降；因此坐标/表示因素表现为 source×representation interaction，不能把单一归一化当作全局修复。

产物：`/data/fzliang/reid-project/g9/e3_source_target/s06_eval/s06_sweep_summary.json`。

### Custom target IMU filter control

D5 固定四个 held-out Custom session 的 AlphaPose tracklets、GT/person order、24/16 protocol 和 EgoHumans source checkpoint，仅替换 aligned 7D IMU 的 quaternion。raw 与“仅替换异常 quaternion（valid 帧不改）”使用完全相同的可见 GT 分母；另跑了全量 unit-normalization 作为敏感性变体。

| session | invalid quaternion fraction | raw history | invalid-fill history | delta correct |
|---|---:|---:|---:|---:|
| 20260211_171423 | 3.197% | 0.43193 | 0.43193 | 0 |
| 20260211_171724 | 0.075% | 1.00000 | 1.00000 | 0 |
| 20260211_172257 | 15.259% | 0.31250 | 0.26745 | −160 |
| 20260211_172522 | 2.257% | 0.40987 | 0.40987 | 0 |
| **aggregate** | — | **0.54714** | **0.53671** | **−160** |

aggregate denominator 为 15,349；history FrameAcc 变化 −0.01042，instantaneous FrameAcc 保持 0.68128 不变。unit-normalized 变体与 invalid-fill 结果相同。结论是 quaternion 异常影响集中在单一 session，且最近有效值填充不是普适修复，不能把 0.91%/局部异常直接解释成全局性能下降。

产物：`/data/fzliang/reid-project/g9/e3_source_target/custom_imu_filter_control.json`。

### S06 prediction strata

D6 将 D3 的 528 个逐序列预测与 S06 输出的 bone-scale-normalized motion energy、visibility coverage 和 visibility-run fragmentation proxy 连接，6 个方法 × 2 个坐标变体全部无缺失。分桶阈值采用 pooled six-method tertiles；fragmentation 不是 ID-switch，因为 S06 没有独立 tracker IDs。

产物：`/data/fzliang/reid-project/g9/e3_source_target/s06_prediction_stratification.json`。

### G6 representation boundary

D7 用相同 xy、随机且显著不同的 z 构造 3D skeleton，直接重算当前 G6 `raw_pose_sequence` 和 `skeleton_tokens`。两者最大绝对差均为 0；因此当前 checkpoint 对 z 完全不敏感。full-xyz attribution 不是本协议中尚未“碰巧漏跑”的 cell，而是必须新建 xyz-compatible encoder/protocol 后才能识别的研究问题。

产物：`/data/fzliang/reid-project/g9/e3_source_target/g6_representation_boundary.json`。

## Missing controlled cells

1. 固定 IMU、固定 Custom target 后的 S06 skeleton-source sweep（已完成；当前是固定检查点、xy 投影诊断）；
2. 2D/3D representation-controlled transfer（当前 G6 协议由 D7 证明 z 不可识别；full-xyz 需新 protocol）；
3. full-xyz representation-controlled transfer；
4. 带 raw prediction 的 complexity/visibility/fragmentation `correct/total` 分层（已完成；ID-switch 仍不可观测）。

剩余缺失项是 full-xyz 表示控制和独立 detector track-ID 审计，而不是被当前 G6 canonical 结果隐含替代。

## Existing prediction strata

`D2_stratify_predictions.py` 已对现有 G6 105 个 session clips 重算 history/instantaneous 两种 assignment 的 `correct/total`，按每个 Custom session 的 low/mid/high target motion tertile、candidate group size 和 visible people 分层；没有缺失 segment。该结果只覆盖现有 canonical G6 runs；S06 sweep 的 6×2 结果见上节，尚未构成重训后的 source-domain 性能排名。
