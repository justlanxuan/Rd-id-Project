# G9 Goal：完成 Source→Custom 骨架 Gap 正确性检查

## 1. 阶段目标

确认 TotalCapture、EgoHumans、Custom 及历史多种骨架源是否在以下语义上正确可比：

- 17 个关节的顺序和 COCO→H36M mapping；
- skeleton person 与 IMU person 的对应关系；
- raw/normalized 坐标空间、2D/3D/SMPL 表示；
- frame/timestamp/window 对齐；
- missing、confidence、tracklet 和异常帧；
- 不同算法目录是否包含真正独立的内容。

## 2. 纳入骨架源

正式检查纳入：TotalCapture GT、EgoHumans pose2d、AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer、WHAM，以及已有 Custom canonical/YOLO raw/AlphaPose raw/WHAM raw 资产。

PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 只有通过独立真实 smoke 后才可加入。

## 3. E1 选择性门禁与可量化完成标准

E1 不是“所有候选源必须同时通过”的全局门禁，而是一个可追溯筛选器。每个 source 独立标记为 `included`、`conditional`、`excluded` 或 `pending`；只有 `included` 源进入第一版正式 gap 矩阵，其他源必须保留排除理由和后续条件。只要至少有一个可信 source、一个可信 Custom target，并且 person/IMU/time join 可重算，G9 就可以在这个最小可信子集上继续。

- 每个纳入 source 有稳定 manifest、provenance、content hash 和表示空间声明；
- 纳入子集的样本通过非空、finite、shape、joint-count、frame-order 检查；
- skeleton/IMU person mapping 有字段证据或明确标记为未验证；
- raw 与 normalized 坐标不能被静默混合；
- S06 同 sequence 多算法输出完成 pairwise content/fingerprint 检查；
- YOLO-Pose high 异常范围完成逐帧/逐关节定位；
- 生成 `semantic_audit.json`、`gap_profile.json` 和 E1 结果表；
- 未满足语义正确性的 source 不进入正式矩阵，但不阻塞已通过子集的 gap 分析。

## 4. 当前状态

`diagnostic_complete_protocol_boundaries_explicit`：E1 选择性门禁、全量语义/坐标/异常审计、IMU/动作复杂度/时间/tracklet screening，以及 S06 六源×raw/screen 固定检查点控制、Custom IMU quaternion 对照、S06 prediction-level 分层和 Custom detector-ID audit 均已完成。full-xyz 与 S06 独立 ID 被证明是当前 G6 协议之外的边界，不再被静默声称完成。

## 5. 验收产物

- `E1:gap-audit/results/results.md`
- `/data/fzliang/reid-project/g9/e1_gap_audit/source_inventory.json`
- `/data/fzliang/reid-project/g9/e1_gap_audit/semantic_audit.json`
- `/data/fzliang/reid-project/g9/e1_gap_audit/gap_profile.json`
- `/data/fzliang/reid-project/g9/e3_source_target/s06_eval/s06_sweep_summary.json`
- `/data/fzliang/reid-project/g9/e3_source_target/custom_imu_filter_control.json`
- `/data/fzliang/reid-project/g9/e3_source_target/s06_prediction_stratification.json`
- `/data/fzliang/reid-project/g9/e3_source_target/g6_representation_boundary.json`
- `/data/fzliang/reid-project/g9/e2_multimodal/custom_detector_id_audit.json`
- `/data/fzliang/reid-project/g9/g9_final_gap_manifest.json`
