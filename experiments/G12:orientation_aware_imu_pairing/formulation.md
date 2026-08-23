# G12 Formulation：人物朝向变化对 IMU 配对的影响

## Need

现有 G10/G11 模型使用 skeleton position、global motion、raw skeleton 和 IMU acceleration/gyro，但没有显式建模人物朝向。IMU 角速度可能携带身体转向信息，也可能主要来自手臂摆动、前臂旋转或佩戴姿态变化。若不把这些因素拆开，朝向信息的作用无法归因。

本 goal 的“现有 skeleton”首先指我们自己的提取器输出。数据集自带 Vicon/SMPL/SMPL-X 朝向只能作为 reference/control；不能因为 source 中存在 `global_orient` 就声称 YOLO-Pose high、AlphaPose、lifting 或 WHAM 的 canonical pair input 已经包含该信息。

## Goal

在统一时间、候选组和 FrameAcc 契约下，回答：

1. 哪些 extractor 的 raw artifact 有直接 root/global orientation，哪些字段在 canonical adapter 中被丢弃？
2. 哪些 extractor 只能从 3D 关节位置推导 torso heading，哪些只有 2D image-plane proxy？
3. body/root yaw rate、torso orientation rate 与 IMU angular velocity 是否存在稳定对应？
4. orientation feature 是否提供超出位置、加速度和已有 gyro 输入的配对信息？
5. wrist gyro 中有多少成分来自 body heading，多少来自 arm-local rotation？
6. 在 orientation 缺失的 Custom canonical 输入上，哪些结论仍然可以成立？

## Hypotheses

- **H1：** TotalCapture 的 root/torso orientation change 与 IMU gyro 存在可测的跨模态关联。
- **H2：** root orientation 或 torso heading 比单纯 2D skeleton motion 更能解释整体转向。
- **H3：** wrist gyro 不能直接作为 body yaw rate，必须保留 sensor-to-body relative rotation 与 arm-swing 混杂解释。
- **H4：** source-domain orientation 增益不能在没有 Custom orientation ground truth 时直接升级为跨域收益。
- **H5：** 2D skeleton 的肩胯方向只能作为弱 proxy；遮挡、视角和深度变化会使其不稳定。

## Scope and exclusions

- 首轮只做 orientation inventory、contract 和物理关联审计；不立即改动正式模型输入。
- root/torso orientation 与局部 segment orientation 分轨，不把两者合并成单一“朝向”标量。
- 不把当前 Custom 2D skeleton 伪装成世界 yaw 真值。
- 不把 raw WHAM 的 `root_orient`/`pose_world` 伪装成当前 canonical `extract_skeleton` 已可用；若要使用，必须新增 adapter 字段与版本化 contract。
- 未通过真实 smoke 的 3D pose/SMPL 后端不进入正式 Custom 结论。
- G12 不回写 G10/G11 的既有 freeze；如需模型训练，使用独立 protocol version。

## Metrics

### Data and physical validity

- orientation availability/validity coverage；
- quaternion norm、sign continuity、axis-angle range 和 yaw wrap failures；
- coordinate-frame and timestamp provenance；
- body heading 与 gyro 的 lagged circular correlation、coherence、互信息或解释方差；
- shuffled-person、time-shift、nonmatching-session 等 null controls。

### Pairing diagnostics

- raw `correct/total` 与 FrameAcc；
- 三 seed `mean ± sample std`；
- positive/negative similarity margin、candidate-group size、singleton rate；
- orientation-missing coverage、参数量、延迟和显存。

## Validity rules

1. 直接 orientation、3D-joint-derived heading、2D proxy 必须分表。
2. 任何 yaw derivative 必须使用真实 timestamp，禁止按文件名或 frame 数假定采样率。
3. quaternion sign、角度 wrap、坐标系和 sensor placement 必须写入 manifest。
4. Source 物理相关性不等于跨域 re-id 改善；两者必须分别报告。
5. Custom orientation 缺失时，正式结果只能报告 orientation-missing/control 或经明确标注的 proxy diagnostic。
