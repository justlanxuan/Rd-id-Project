# E1 Progress

## 2026-08-18

- 已创建 G9 Formulation、Plan 和 E1 审计沙盒。
- 已登记 TotalCapture GT、EgoHumans pose2d、AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer、WHAM。
- 已将 PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 标记为待真实 smoke 的候选。
- 新增并运行 `scripts/A1_build_source_inventory.py`，输出 `/data/fzliang/reid-project/g9/e1_gap_audit/source_inventory.json`。
- 14 个 source/cache 入口均存在：TotalCapture GT、EgoHumans canonical、Custom canonical、S06 六种算法、Custom YOLO raw、AlphaPose raw、WHAM raw、两组 EgoHumans pose2d cache。
- 抽样文件均为 finite；抽样 skeleton fingerprint 未发现 exact duplicate group。
- 初步发现 EgoHumans canonical raw 坐标量级与 Custom/S06 normalized 坐标不同；S06 YOLO-Pose high 的 sample bone CV/范围异常，需优先审计。
- 用户确认 E1 采用选择性门禁：少量正确骨架即可进入 gap 分析，不要求所有候选源通过。
- 新增并运行 `scripts/A2_semantic_skeleton_audit.py`：TotalCapture/EgoHumans canonical 样本的 person/IMU 显式相等；Custom 四个 fold 共 12 个 window CSV、7380 行、7380 个 NPZ 引用全部映射一致且文件存在；S06 train/val 共 108 个 baseline 的 person/IMU 映射全部一致；S06 同序列 15 对输出无 exact duplicate。
- 当前最小可信子集：TotalCapture GT、EgoHumans canonical、Custom canonical、AlphaPose、FMPose3D、MotionAGFormer、TCPFormer、WHAM。YOLO-Pose high 为 conditional（覆盖率/坐标异常待查），raw cache 为 pending。
- 新增并运行 `scripts/A3_build_gap_profile.py`，生成 `/data/fzliang/reid-project/g9/e1_gap_audit/gap_profile.json`，记录坐标、身份/时间、质量/独立性证据和四个后续可检验假设。
- 尚未运行新的骨架提取、预处理或训练。

## Next actions

1. 扩展 E1 content fingerprint 到同 sequence 多算法和多序列；
2. 审计 EgoHumans raw/normalized 与 YOLO-Pose high outlier；
3. 在最小可信子集上生成 gap profile，保留 conditional/pending 源供后续诊断；
4. 完成 E1 坐标空间与 coverage 记录后拟定 E2 正式矩阵。
