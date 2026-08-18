# E3 Results：Source/Target 矩阵索引

状态：`indexed_existing_only`。这一步没有把未运行的 S06 算法伪装成 benchmark；它只索引已有 G6 结果并明确缺失控制条件。

产物：`/data/fzliang/reid-project/g9/e3_source_target/source_target_matrix.json`。

- G6 已有 66 个 completed evaluation records，聚合为 22 个 source/condition/session cells，保留每个 seed 的 `correct/total`、protocol hash 和 run-record 路径。
- 已有 source performance 只覆盖 canonical TotalCapture 和 EgoHumans；AlphaPose、FMPose3D、MotionAGFormer、TCPFormer、WHAM 目前只有 E1/E2 语义/运动证据，没有 G6 性能 cell。
- YOLO-Pose high 明确为 conditional，未进入任何正式性能排名。
- 四个 Custom session 的 target motion/IMU 特征已附到每个 G6 cell；zero-shot 的 target motion-energy 与 FrameAcc 相关性仅作描述性探索（每个 source 只有 n=4），不作因果结论。

## Missing controlled cells

1. 固定 IMU、固定 Custom target 后的 S06 skeleton-source sweep；
2. 2D/3D representation-controlled transfer；
3. 统一 7D IMU contract、Custom invalid quaternion filtering 后的 IMU-only/skeleton-only/fusion 对照；
4. 带 raw prediction 的 complexity/tracklet `correct/total` 分层。

这些缺失项是 G9 完成前的必要工作，而不是被当前 G6 canonical 结果隐含替代。

## Existing prediction strata

`D2_stratify_predictions.py` 已对现有 G6 105 个 session clips 重算 history/instantaneous 两种 assignment 的 `correct/total`，按每个 Custom session 的 low/mid/high target motion tertile、candidate group size 和 visible people 分层；没有缺失 segment。该结果只覆盖现有 canonical G6 runs，S06 skeleton-source sweep 仍待执行。
