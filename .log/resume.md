# HAROS Task Resume Node

## Current Stage

G9 Plan：Source-to-Custom 骨架与跨模态域差异分解；E1 选择性门禁完成，当前处于可信子集的 E2/E4/E5 screening 阶段。

## Latest achievements

- G6 的 42 个训练和 66 个评估单元已在独立 artifact root 完成并验证。
- G9 已获人类确认，正式纳入 TotalCapture GT、EgoHumans pose2d、AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer 和 WHAM。
- PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 暂列为待真实 smoke 候选。
- G9 Formulation、Plan 和 E1 测试契约已写入仓库。
- G9 起始锚点已提交：`6c86edd`。
- E1 A1 sample inventory 已运行，14 个入口存在，抽样数据 finite，未发现 exact duplicate fingerprint group。
- E1 A2 语义审计已运行：Custom 四 fold 的 7380 行窗口映射一致；S06 train/val 108 个 baseline 的 person/IMU 映射一致；S06 同序列 15 对算法无 exact duplicate。
- 已按 source 输出 `included/conditional/pending`：当前最小可信子集为 TotalCapture GT、EgoHumans canonical、Custom canonical、AlphaPose、FMPose3D、MotionAGFormer、TCPFormer、WHAM；YOLO-Pose high 暂为 conditional。
- 已生成 `/data/fzliang/reid-project/g9/e1_gap_audit/gap_profile.json`，将坐标、时间/身份、质量和跨模态因素拆成可验证假设。
- E2/E4/E5 B1 全量扫描完成：46 TotalCapture、30 EgoHumans、7380 Custom windows、5 个 S06 方法各 88 序列；输出 `multimodal_motion_diagnostics.json`。
- B2 完成 S06 coverage/tracklet/fragmentation 与 baseline visibility 对照；独立 ID switch 因输出继承 GT person order 且无独立 track IDs，标记为不可识别。
- 发现关键 representation gap：EgoHumans/AlphaPose/YOLO-Pose 是 2D xy+visibility/zero-z，不能按 `[... ,3]` 当作 3D；S06 legacy48 IMU 的选定 L_LowArm acceleration energy 中位数约 22.8，而 canonical 7D 流约 125–129，需先统一 IMU contract。
- C1 已将 S06 legacy48 转换到显式 7D contract；同时发现 Custom 有 1612/177120（0.91%）invalid quaternion frames（28 个 zero-norm），因此 Custom skeleton-target 可纳入，但 IMU/fusion 需报告过滤分母。
- A4 已定位 YOLO-Pose high 的 996 个 `abs>10` 坐标（54/88 序列），包括 `custom_01_003` frame 184/person 0/joint 10/y=-30.47；EgoHumans raw xy 约 94.9% 坐标值绝对值大于 10，进一步确认必须分 representation/coordinate track。

## Blockers / Issues

- G9 source inventory 已建立为 sample-level；全量 content fingerprint、逐关节 outlier 和 coverage manifest 尚未完成。
- 未验证的骨架后端不能进入正式结果；但未通过的候选不会阻塞可信子集上的 gap 分析。
- 当前 GPU 资源被其他进程使用；大型 artifact 必须写入 `/data`。

## Next actions

1. 审计 EgoHumans raw/normalized 坐标与 YOLO-Pose high outlier；
2. 将可信 IMU 流统一到显式 7D contract，重跑 lag/复杂度统计；
3. 接入正式预测结果，生成复杂度/tracklet 分层 `correct/total`；
4. 冻结 G9 protocol version 后启动 IMU-only、skeleton-only、fusion 和 skeleton-source sweep。
