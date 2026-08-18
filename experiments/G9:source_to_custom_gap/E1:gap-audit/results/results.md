# E1 Results：骨架源与输入 gap 审计

状态：`in_progress`（选择性门禁的最小可信子集已建立；全量内容指纹、坐标空间和 outlier 归因仍待完成）

审计产物：`/data/fzliang/reid-project/g9/e1_gap_audit/source_inventory.json`

本轮使用 `--sample-limit 4 --full-hash --max-npz-inspect-mb 64`。`--full-hash` 表示对抽样文件计算完整 SHA256；尚未对每一个大型 artifact 做全量 hash，因此不能将本轮标记为最终 E1 完成。

语义审计产物：`/data/fzliang/reid-project/g9/e1_gap_audit/semantic_audit.json`。本轮门禁按 source 独立决策，不要求所有候选骨架同时通过。

### Selective E1 gate

| Source | Status | Scope | Evidence / limitation |
|---|---|---|---|
| TotalCapture GT | included | canonical GT→IMU gap | sample `[4115,1,17,3]`，`gt_person_ids == imu_ids`，frame/IMU 长度一致 |
| EgoHumans canonical | included | canonical GT→IMU gap | sample `[601,4,17,3]`，person/IMU 显式相等；raw 与 normalized 坐标仍需分轨 |
| Custom canonical | included | target Custom | 四 fold、12 个 window CSV、7380 行、7380 个 NPZ 引用；`person_idx == imu_idx`，无缺失引用，source=`gt` |
| S06 AlphaPose | included | skeleton + external IMU join | 输出含 `gt_person_ids`；108 个 train/val baseline 全部验证 `gt_person_ids == imu_ids` |
| S06 FMPose3D | included | skeleton + external IMU join | 同上；不与其他 3D 输出视为 exact duplicate |
| S06 MotionAGFormer | included | skeleton + external IMU join | 同上；与 FMPose3D/TCPFormer 高相关但数值不完全相同 |
| S06 TCPFormer | included | skeleton + external IMU join | 同上 |
| S06 WHAM | included | skeleton + external IMU join | 同上 |
| S06 YOLO-Pose high | conditional | diagnostic only | 覆盖率较低且抽样 bone CV/范围异常；待逐帧坐标/outlier 审计 |
| raw pose/WHAM/cache artifacts | pending | diagnostic only | 尚未提供 canonical person/IMU/time join |

因此当前可继续 gap 分析的最小可信子集为：`TotalCapture GT`、`EgoHumans canonical`、`Custom canonical`、`S06 AlphaPose`、`S06 FMPose3D`、`S06 MotionAGFormer`、`S06 TCPFormer`、`S06 WHAM`。`conditional`/`pending` 源不会阻塞该子集，也不能进入第一版正式 source→Custom 主矩阵。

### Semantic audit evidence

| Check | Result |
|---|---|
| Custom all-fold mapping | `verified_equal`；12 CSV、7380 rows、7380 unique NPZ refs、0 mismatch、0 missing |
| S06 baseline mapping | `verified_equal`；train/val 2 manifests、108 sequences、0 missing、0 mismatch |
| S06 same-sequence pairwise | 15 pairs，0 exact duplicate；3D 方法间高相关但非相同张量 |
| Canonical frame/IMU counts | TotalCapture/EgoHumans/Custom sample 均相等；Custom timing 由 window CSV 提供，NPZ 无 embedded frame_ids |

## Required tables

### Source inventory

| Dataset/source | Representation | Files | Sample schema/quality | Status |
|---|---|---:|---|---|
| TotalCapture GT | 3D, 17 joints | 46 NPZ + 46 JSON | finite；`[T,1,17,3]`；sample bone CV 0.477 | included |
| EgoHumans canonical | 3D, 17 joints | 30 NPZ + 30 JSON | finite；`[T,4,17,3]`；raw coordinate scale需确认 | included, normalization audit required |
| Custom G6 canonical | 2D, 17 joints | 7380 NPZ + 4 JSON + 12 CSV | finite；`[24,17,2]`；sample bone CV 0.480 | included |
| S06 AlphaPose | unified 3D, 17 joints | 88 NPZ | finite；`[T,4,17,3]`；sample bone CV 0.532 | included |
| S06 YOLO-Pose high | unified 3D, 17 joints | 88 NPZ | finite；sample range约 `[-30.47,12.09]`；bone CV 1.355 | included, outlier audit required |
| S06 FMPose3D | unified 3D, 17 joints | 88 NPZ | finite；sample bone CV 0.489 | included |
| S06 MotionAGFormer | unified 3D, 17 joints | 88 NPZ | finite；sample bone CV 0.513 | included |
| S06 TCPFormer | unified 3D, 17 joints | 88 NPZ | finite；sample bone CV 0.474 | included |
| S06 WHAM | unified 3D, 17 joints | 88 NPZ | finite；sample bone CV 0.436 | included |
| Custom YOLO-Pose raw | 2D/raw canonical artifacts | 57 NPZ + 114 JSON | raw output inventory available | included |
| AlphaPose raw results | COCO-17 JSON + visualizations | 7 JSON + 11882 JPG + 9 MP4 | raw detector inventory available | included |
| WHAM raw output2 | SMPL/mesh PKL/PTH | 7 PKL + 14 PTH | raw 3D/SMPL inventory available | included |
| EgoHumans pose2d cache (sync) | 2D NPY | 1540 NPY | cache inventory available | included |
| EgoHumans pose2d cache (w24) | 2D NPY | 13427 NPY | cache inventory available | included |

### Quality summary

| Source | Sample coordinate dim | Finite | Bone CV | Initial note |
|---|---:|---:|---:|---|
| TotalCapture GT | 3 | 1.0 | 0.477 | reference scale |
| EgoHumans canonical | 3 | 1.0 | 0.866–0.971 | raw coordinate magnitude much larger，需核对 normalization |
| Custom canonical | 2 | 1.0 | 0.480 | normalized 2D sample |
| S06 AlphaPose | 3 | 1.0 | 0.532 | finite |
| S06 YOLO-Pose high | 3 | 1.0 | 1.355 | large coordinate outlier，需逐关节审计 |
| S06 FMPose3D | 3 | 1.0 | 0.489 | finite |
| S06 MotionAGFormer | 3 | 1.0 | 0.513 | finite |
| S06 TCPFormer | 3 | 1.0 | 0.474 | finite |
| S06 WHAM | 3 | 1.0 | 0.436 | finite |

## Initial interpretation

1. 所有 14 个入口均存在，S06 六种算法各有 88 个 NPZ；这确认了候选资产的实际可用性，但不等同于每个后端已在当前机器重新生成。
2. 本轮抽样 skeleton fingerprint 未发现跨 source 的 exact duplicate group；仍需扩大到同 sequence、多序列 fingerprint 才能排除重复产物。
3. EgoHumans canonical 与 Custom/S06 的坐标量级明显不同，必须同时保存 raw/normalized schema，不能直接用 raw 数值距离做 domain gap 结论。
4. YOLO-Pose high sample 的 bone CV 和坐标范围异常大，优先进入 E1 outlier audit；在确认是否为合法坐标空间前，不做算法排名。
5. E1 后续必须补充逐文件 hash、missing/confidence/tracklet 汇总和 source/Custom 可配对 sequence 覆盖率。
6. “finite” 只证明数值可计算，不证明关节语义、身份或时间对齐；本轮 A2 才开始建立这些语义证据。
