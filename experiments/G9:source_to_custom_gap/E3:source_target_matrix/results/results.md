# E3 Results：Source/Target 矩阵索引

状态：`indexed_plus_s06_fixed_checkpoint_control`。已有 G6 结果仍按原协议索引；另外完成了 S06 六种骨架源在固定 G6 检查点上的全量诊断扫掠。

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

## Missing controlled cells

1. 固定 IMU、固定 Custom target 后的 S06 skeleton-source sweep（已完成；当前是固定检查点、xy 投影诊断）；
2. 2D/3D representation-controlled transfer（仍需支持 full-xyz encoder 或重训）；
3. 统一 7D IMU contract、Custom invalid quaternion filtering 后的 IMU-only/skeleton-only/fusion 对照；
4. 带 raw prediction 的 complexity/tracklet `correct/total` 分层。

这些缺失项是 G9 完成前的必要工作，而不是被当前 G6 canonical 结果隐含替代。

## Existing prediction strata

`D2_stratify_predictions.py` 已对现有 G6 105 个 session clips 重算 history/instantaneous 两种 assignment 的 `correct/total`，按每个 Custom session 的 low/mid/high target motion tertile、candidate group size 和 visible people 分层；没有缺失 segment。该结果只覆盖现有 canonical G6 runs；S06 sweep 的 6×2 结果见上节，尚未构成重训后的 source-domain 性能排名。
