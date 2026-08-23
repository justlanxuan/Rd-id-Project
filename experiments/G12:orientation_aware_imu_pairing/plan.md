# G12 Plan：人物朝向感知的 IMU–骨架配对探索

## 0. 冻结边界

- Goal：`G12:orientation_aware_imu_pairing`。
- 研究对象：body/root orientation、torso heading、local segment orientation 与 IMU angular velocity 的关系。
- 直接 orientation、3D-derived heading、2D proxy、orientation-missing 必须分轨。
- 主审计对象是我们自己的 extractor 输出（尤其 `/data/lyxie/ReID/Pipeline/Skeleton_Extractors`、S06 source-ablation 与 YOLO-Pose high artifacts），而不是数据集原生骨架的朝向。
- TotalCapture Vicon/SMPL-X 与 EgoHumans full SMPL 只作为 source-orientation reference/control；不能替代 extractor-level schema 审计。
- 当前 canonical extractor cache 只有关节位置：2D 来源是 image-plane proxy，3D 来源可派生 heading；WHAM raw 虽有 `root_orient`/`pose_world`，但现有 adapter 未把它传入 `extract_skeleton`。
- 不回写 G10/G11 freeze；模型训练需新建 G12 protocol version。
- 大型 artifact 写入 `/data/fzliang/reid-project/g12/`。

## 1. E1：Skeleton orientation inventory（extractor-first）

### 目标

建立所有实际 extractor artifact 的字段、shape、关节语义、坐标系、时间和 provenance inventory；同时记录 orientation 是否在 canonical pair input 中仍然存在。

### 数据入口

| 来源 | 角色 | 首要字段 |
|---|---|---|
| YOLO-Pose high | 2D extractor output | `pipeline_extract.npz`, `extract_keypoints_coco17`, `extract_skeleton` |
| AlphaPose | 2D extractor output | `skeleton.json`, `algorithm_outputs/alphapose/*.npz` |
| FMPose3D | 3D lifting output | `skeleton`, metadata `root-centered torso-scaled H36M17 xyz` |
| MotionAGFormer | temporal 2D→3D output | `skeleton`, same root/scale contract |
| TCPFormer | temporal 2D→3D output | `skeleton`, same root/scale contract |
| WHAM | raw SMPL/world reconstruction + canonical joints | raw `root_orient`/`pose_world`; canonical `skeleton` |
| Dataset-native orientation reference | control only | Vicon/SMPL-X/SMPL fields from prior inventory |

### 任务

1. 抽样并统计文件数量、shape、dtype、finite、visibility 和时间覆盖。
2. 检查 canonical `extract_skeleton` 是否仍含 rotation/root orientation；不把 raw WHAM 字段误报为模型已使用。
3. 对 2D、3D joint-derived、raw-direct-but-not-canonical、direct-propagated 分轨，不以文件名推断语义。
4. 对候选 artifact 生成稳定内容 fingerprint，不纳入绝对路径和 mtime。
5. 输出 orientation coverage、坐标系待确认项和不能进入正式表的原因；source inventory 作为附录 control。

### Gate A

- 没有 object-array 误读、空文件、NaN 或 quaternion norm 异常；
- 字段、时间和人/序列 identity 可追溯；
- direct 与 derived/proxy 分类可复核；
- extractor raw 到 canonical pair input 的字段丢失链路可复核；
- 所有失败样本原样记录，不用空结果代替。

## 2. E2：Orientation contract and derivation

E2 contract 已冻结为 `g12.orientation_contract.v1`，实现位于 `src/features/orientation.py`，机器可读 manifest 位于 `E2:orientation_contract/results/contract_manifest.json`。该 contract 只产生独立 feature object，不回写正式 canonical schema；E2 tests 通过前不运行 orientation-aware pairing 训练。

冻结候选字段：

```yaml
root_orientation_quat
root_orientation_axis_angle
root_orientation_6d
root_yaw_sin_cos
root_yaw_rate
torso_forward
segment_orientation
sensor_to_body_relative_rotation
orientation_valid
orientation_source
coordinate_frame
```

必须建立 quaternion sign continuity、axis-angle conversion、yaw wrap、真实 timestamp differentiation、3D-joint heading degeneracy 和 missingness tests。E2 输出 contract，不立即修改正式 preprocess schema。

## 3. E3：Physical orientation–gyro audit

E3 corrected audit 已完成。骨架不需要包含 gyro：六种 S06 algorithm output 的 orientation track 均按 `custom_XX_YYY + sorted aria person + frame index` 与独立 EgoHumans realistic-IMU LeftWrist gyro 对齐。每种方法覆盖 88 sequences/313 person tracks，join failures=0。比较：

- body/root yaw rate ↔ IMU gyro xyz/magnitude；
- torso heading rate ↔ sensor-local angular velocity；
- pelvis/torso/left-arm 不同 anchor；
- lagged correlation、coherence、互信息、频带能量和解释方差；
- shuffled person、time shift、nonmatching session、arm-only movement 控制。

输出必须按 dataset、sensor placement、orientation source 和 motion stratum 分层，并记录 measured-vs-kinematic gyro provenance。

机器结果：[corrected E3 audit artifact](/data/fzliang/reid-project/g12/e3_physical_audit/extractor_orientation_imu_join_corrected.json)；脚本与结果说明位于 `E3:physical_orientation_gyro_audit/`。旧 artifact 中“extractor 文件本身缺少 gyro 所以不可分析”的判断已作废。本阶段未运行 pairing ablation。

## 4. E4：Orientation-aware pairing ablation

E3 完成后已进入首轮 controlled training。实现和 raw metrics 归档在 `E4:orientation_model_validation/`；首轮不改变正式 G11 schema。

已运行：

1. 现有 skeleton + IMU baseline；
2. baseline + root yaw sin/cos；
3. baseline + root/torso yaw rate；
4. baseline + 6D orientation；
5. baseline + segment orientation；
6. baseline + sensor-to-body relative rotation。

首轮主 screen 聚焦 baseline vs extractor-derived 2D proxy + turning gate；concat/gyro-focus/residual 仅作实现级 sanity controls。0.8 秒三 seed 没有稳定收益；2 秒出现高方差 Custom23 信号，暂不 promotion。3D heading 和 sensor-to-body relative rotation 仍需后续独立 protocol，不能从 2D proxy 结果推断。

固定 candidate-group、window、normalization、source regime 和 seed 数量。至少三 seed；source accuracy 只作诊断，不能替代 target protocol。Custom 若仍没有 direct/validated orientation，只运行 orientation-missing/proxy control，不进入正式 orientation promotion。按 Custom session 语义，23 是转向收益目标；57/22/24 是非转向负控，不应被要求获得 orientation gain。

## 5. E5：Protocol and conclusion gate

正式汇总至少包含：

- raw `correct/total`、FrameAcc、margin 和 candidate-group size；
- orientation coverage、missingness、坐标系和 provenance；
- source/session 分层与三 seed `mean ± sample std`；
- 参数量、延迟、显存和 artifact hashes；
- 哪些结论是物理相关性，哪些是 pairing gain，哪些因 Custom 缺失 orientation 而不能判断。

若后续需要把 orientation 加入 canonical schema 或 Custom 3D backend，必须新建 protocol version 并重新审计，不得把 E4 结果直接升级 G11/G10 freeze。

## 6. 阶段验收

| 阶段 | 通过标准 |
|---|---|
| E1 | inventory 完整、direct/derived/proxy/missing 分类可复核 |
| E2 | conversion、continuity、wrap、timestamp derivative 和 missingness 契约测试通过 |
| E3 | extractor-derived/raw orientation–gyro 物理审计含 hard negatives 和 provenance 分层 |
| E4 | orientation ablation 至少三 seed，保留 raw counts 和完整 artifact identity |
| E5 | 汇总不把 source 相关性、raw-only WHAM、2D proxy 或 orientation-missing 误报为 Custom 正式收益 |
