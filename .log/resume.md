# HAROS Task Resume Node

## Current Stage

G9 Plan：Source-to-Custom 骨架与跨模态域差异分解；当前处于 E1 骨架源资产审计准备阶段。

## Latest achievements

- G6 的 42 个训练和 66 个评估单元已在独立 artifact root 完成并验证。
- G9 已获人类确认，正式纳入 TotalCapture GT、EgoHumans pose2d、AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer 和 WHAM。
- PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 暂列为待真实 smoke 候选。
- G9 Formulation、Plan 和 E1 测试契约已写入仓库。

## Blockers / Issues

- 尚未建立 G9 source inventory、content fingerprint 和 provenance manifest。
- 未验证的骨架后端不能进入正式结果。
- 当前 GPU 资源被其他进程使用；大型 artifact 必须写入 `/data`。

## Next actions

1. 完成 E1 source inventory 和质量审计；
2. 检查 S06 多算法输出是否内容独立；
3. 通过 E1 测试门后冻结 G9 protocol version；
4. 再启动 IMU-only、skeleton-only、fusion 和 skeleton-source sweep。
