# E1 Progress

## 2026-08-18

- 已创建 G9 Formulation、Plan 和 E1 审计沙盒。
- 已登记 TotalCapture GT、EgoHumans pose2d、AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer、WHAM。
- 已将 PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 标记为待真实 smoke 的候选。
- 新增并运行 `scripts/A1_build_source_inventory.py`，输出 `/data/fzliang/reid-project/g9/e1_gap_audit/source_inventory.json`。
- 14 个 source/cache 入口均存在：TotalCapture GT、EgoHumans canonical、Custom canonical、S06 六种算法、Custom YOLO raw、AlphaPose raw、WHAM raw、两组 EgoHumans pose2d cache。
- 抽样文件均为 finite；抽样 skeleton fingerprint 未发现 exact duplicate group。
- 初步发现 EgoHumans canonical raw 坐标量级与 Custom/S06 normalized 坐标不同；S06 YOLO-Pose high 的 sample bone CV/范围异常，需优先审计。
- 尚未运行新的骨架提取、预处理或训练。

## Next actions

1. 扩展 E1 content fingerprint 到同 sequence 多算法和多序列；
2. 审计 EgoHumans raw/normalized 与 YOLO-Pose high outlier；
3. 生成 E1 最终 manifest 与质量报告；
4. 通过 E1 测试门后再拟定 E2 正式矩阵。
