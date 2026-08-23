# E4 Results：orientation-aware matcher

## 实现

`src/g12/orientation_motion.py` 从 extractor skeleton 的原始 2D xy 派生 torso shoulder-line proxy，并输出 `sin_angle, cos_angle, rate_scaled, orientation_valid, turning_activity`。`src/g12/orientation_matcher.py` 保留 G11 的 skeleton/IMU TemporalEncoder；turning-gate 额外编码 orientation，并以 learned sigmoid gate 注入 skeleton embedding。IMU gyro 仍来自独立 sidecar，未写入 skeleton。

代码 fingerprint：`orientation_motion.py`=`475c9c82c689541cefaea636d5516d66f9efc29f6ec700e1beb86c2defda9458`；`orientation_matcher.py`=`bbfdca3e9a1ff6213a81c9c732fbcc9f41ad2055a9e79654a22105659f085d89`；训练器=`ea7bb763148b7cb9acee2b4b7aa0205d677800aef52f281e59e42ceea44475c4`。

## Custom skeleton provenance 与转向分层

Custom corrected cache 的 `skeleton` 与 `extract_skeleton` 都是 AlphaPose 提取后转成 H36M17 的 2D 关节点：形状为 `[T,2,17,3]`，前两维是 image-plane `(x,y)`，第三维是 AlphaPose keypoint confidence，不是 z 坐标、quaternion 或人物世界朝向。`skeleton` 与 `extract_skeleton` 在四个 corrected session 中逐元素一致；模型 skeleton 主分支最终只使用 `xy+joint_visibility`。

因此本 E4 并没有使用“Custom 自带的直接方向字段”。turning-gate 在 eval 时从同一 Custom 2D AlphaPose skeleton 的左右肩线派生了一个 **2D torso orientation proxy**；它不是经过标定的世界 yaw。Custom23 是预定义的全局大动作/转向 session，才是 orientation gain 的主要目标；57（原地小动作）、22（原地大动作）和 24（静态）应作为非转向负控，不能要求它们获得同等收益。按实际 proxy stream 的 0.8s window 统计：23 的 mean turning activity=`0.1072`、95th percentile=`0.4729`；57=`0`、22=`0.0036`、24=`0.0218`。22/24 的少量 rate 主要应解释为 2D 观测噪声或局部姿态变化，而非已验证的人体整体转向。

## Source 同源性检查（新增限制）

E4 的 train manifests 明确写着 `skeleton_source=gt`。`RawMotionDataset`/`OrientationMotionDataset` 对训练 source 优先读取 `gt_skeleton`；TotalCapture/EgoHumans source 当前并不是用 Custom 侧 AlphaPose extractor 生成的 skeleton。相反，Custom23 eval 读取的是 AlphaPose `skeleton`/`extract_skeleton`。因此当前 E4 只能说明“GT-source encoder + Custom AlphaPose-derived proxy”的初步信号，不能称为同一 extractor 的 source→Custom23 验证。

按建议，下一轮应先固定一个 extractor（默认优先 AlphaPose，因为 Custom23 已有完整 AlphaPose artifact），对 source train 和 Custom23 都读取该 extractor 的 skeleton，再从同一坐标/关节 contract 派生 orientation，最后只在 Custom23 上检验 turning-aware inference。若改用 YOLO-Pose high、FMPose3D 或 WHAM，则必须分别生成 source/Custom23 同源 artifact，不得把不同 extractor 混成一张表。

## 三 seed screen（按 Custom23 最佳 epoch 选择）

### 0.8 秒

| variant | Custom23 | Custom57 | Custom22 | Custom24 |
|---|---:|---:|---:|---:|
| baseline | `0.5185 ± 0.0219` | `0.5000 ± 0.0030` | `0.5000 ± 0.0034` | `0.5032 ± 0.0032` |
| turning-gate | `0.5142 ± 0.0178` | `0.4990 ± 0.0017` | `0.4795 ± 0.0327` | `0.4968 ± 0.0110` |

在 G11 primary 0.8 s protocol 下，显式 turning stream 没有稳定提升；但由于只有 Custom23 是预定义转向 session，57/22/24 的不提升不构成反证。正确的结论是：Custom23 的均值反而低 `0.43` 个百分点，当前 2D proxy 尚未改善真正的转向目标；非转向 session 只能作为不应恶化的负控。不能把 orientation proxy 直接 promotion 为当前短窗模型的默认输入。

### 2.0 秒探索性窗口

| variant | Custom23 | Custom57 | Custom22 | Custom24 |
|---|---:|---:|---:|---:|
| baseline | `0.5041 ± 0.0041` | `0.5000 ± 0.0031` | `0.5000 ± 0.0035` | `0.4968 ± 0.0032` |
| turning-gate | `0.5372 ± 0.0286` | `0.4990 ± 0.0018` | `0.5000 ± 0.0000` | `0.4905 ± 0.0114` |

2 秒窗口的 Custom23 均值提升约 `+3.31` 个百分点，但主要来自 seed 0/1 的 `134/242=0.5537`，seed 2 仅 `0.5041`；sample std 也从 `0.0041` 增至 `0.0286`。57/22/24 是非转向负控，不能作为 turning gain 的必要条件；其中 24 下降约 `0.63` 个百分点仍提示没有普适改善。这应记录为“窗口长度与 turning 信息存在交互的高方差探索信号”，而非稳定模型收益。

## 诊断与结论

- 2 秒 turning-gate 的 gate mean 在各 epoch/seed 约 `0.38–0.48`，说明分支被实际使用，但 gate 不是因果证据；它可能只是在补偿窗口长度或 domain noise。
- `turning_gyro`、`turning_residual`、`turning_concat` 的单 seed sanity 未显示可靠优于 gate；gyro-focus 甚至接近 collapse，因此不进入主结论。
- E3 的 matched orientation–gyro 相关性已被模型实验部分转化为可训练信号，但在当前 0.8 秒正式 protocol 下没有 matching gain。2 秒结果支持继续研究 turning-aware temporal aggregation/辅助 loss，而不支持修改 G11 primary freeze。
- 下一步应在固定 2 秒窗口上增加 turning-stratified evaluation（低/高转向）、更长训练和预注册 seed；同时优先验证 3D-derived heading 与 sensor-to-body frame，而不是继续扩大 2D proxy 的自由度。

完整 raw metrics 与 checkpoints：`/data/fzliang/reid-project/g12/e4_orientation_model/screen_*` 和 `screen_2s_*`。
