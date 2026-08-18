# HAROS Task Resume Node

## Current Stage

G9 Plan：Source-to-Custom 骨架与跨模态域差异分解；当前处于 E1 骨架源资产审计准备阶段。

## Latest achievements

- G6 的 42 个训练和 66 个评估单元已在独立 artifact root 完成并验证。
- G9 已获人类确认，正式纳入 TotalCapture GT、EgoHumans pose2d、AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer 和 WHAM。
- PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 暂列为待真实 smoke 候选。
- G9 Formulation、Plan 和 E1 测试契约已写入仓库。
- G9 起始锚点已提交：`6c86edd`。
- E1 A1 sample inventory 已运行，14 个入口存在，抽样数据 finite，未发现 exact duplicate fingerprint group。

## Blockers / Issues

- G9 source inventory 已建立为 sample-level；全量 content fingerprint、逐关节 outlier 和 coverage manifest 尚未完成。
- 未验证的骨架后端不能进入正式结果。
- 当前 GPU 资源被其他进程使用；大型 artifact 必须写入 `/data`。

## Next actions

1. 扩展 E1 content fingerprint 到同 sequence 多算法和多序列；
2. 审计 EgoHumans raw/normalized 坐标与 YOLO-Pose high outlier；
3. 通过 E1 测试门后冻结 G9 protocol version；
4. 再启动 IMU-only、skeleton-only、fusion 和 skeleton-source sweep。
