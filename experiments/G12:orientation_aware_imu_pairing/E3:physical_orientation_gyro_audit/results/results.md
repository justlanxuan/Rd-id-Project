# E3 Corrected Results：Extractor 朝向变化 × 外部 IMU gyro

## 更正

首轮 E3 错把“骨架 NPZ 没有 gyro 字段”当成无法分析。该判断错误：骨架只负责提供人物朝向；角速度应从独立 IMU 数据读取，再按 sequence/person/frame 对齐。旧 `orientation_gyro_audit.json` 的 extractor-missing-gyro 解释已被本结果取代。

corrected artifact：`/data/fzliang/reid-project/g12/e3_physical_audit/extractor_orientation_imu_join_corrected.json`；SHA256=`d84f33e9fb6e77d016d973dafc57794c27655c8902742454d1f3e4c00669c9ed`

## Join contract

- skeleton：S06 `algorithm_outputs/<extractor>/*.npz`，每种方法 88 个真实算法输出，不重复读取 P25/P50/P100 slice symlink；
- IMU：`/data/lyxie/ReID_imu_generation/outputs/datasets/egohumans/realistic/extracted_data/XX_YYY_aria*.npy`；
- sequence：`custom_XX_YYY → XX_YYY_aria*.npy`；
- person：严格复用原 `convert_realistic_to_pipeline.py` 的 sorted aria file order；
- frame：frame index 一一对应且长度必须完全相同，时间由 raw metadata `target_fps=20 Hz` 构造；
- sensor：LeftWrist gyro，单位 rad/s，provenance 为 SMPL-kinematic realistic IMU，不冒充硬件实测。

全量结果为 6 methods × 88 sequences × 313 person tracks = 1,878 tracks，join failures=0。

## 主要结果

下表是 0.25 秒平滑后，朝向变化率与三轴 gyro 中最大绝对 Pearson 相关的 track 中位数。三轴取最大也会增加乐观偏差，因此必须与完全相同计算方式的 shuffled-person control 一起看。

| Extractor | 朝向类型 | Matched median | Shuffled-person median | Matched 更高比例 | `|yaw-rate|`–gyro norm median |
|---|---|---:|---:|---:|---:|
| YOLO-Pose high | 2D shoulder-line proxy | 0.254 | 0.131 | 81.7% | 0.242 |
| AlphaPose | 2D shoulder-line proxy | 0.198 | 0.119 | 78.0% | 0.145 |
| FMPose3D | 3D torso heading | 0.369 | 0.135 | 91.7% | 0.264 |
| MotionAGFormer | 3D torso heading | 0.456 | 0.151 | 94.2% | 0.325 |
| TCPFormer | 3D torso heading | 0.393 | 0.138 | 92.0% | 0.284 |
| WHAM canonical joints | 3D torso heading | **0.507** | 0.152 | **95.5%** | **0.366** |

time-shuffle null 的 track-level `|r|` 95th percentile 中位数仅为 `0.079–0.090`，显著低于六种方法的 matched 中位数。3D 方法最佳 lag 的中位绝对值为 `0.05–0.10 s`；2D proxy 为 `0.25–0.35 s`，与 3D heading 相比时间对应更弱。

## 结论

1. 人物骨架的朝向变化和对应人物的 LeftWrist gyro 确实存在可观测关系；matched-person 明显高于 shuffled-person/time-shuffle null。
2. 3D-derived torso heading 的关系普遍强于 2D shoulder-line proxy；当前六种输出中 WHAM canonical joints-derived heading 最强，其次为 MotionAGFormer。
3. 关系不是等价关系：wrist gyro 同时包含前臂/手腕局部旋转，extractor heading 又处于未完全校准的 camera/root-centered 坐标系。结果支持把 orientation 作为 pairing 候选信息，但不支持直接将某一 gyro axis 当作世界 yaw-rate。
4. EgoHumans gyro 是由 SMPL 运动学生成，与 pose 共享生成来源，可能高估真实硬件场景关联。进入 E4 前应在 TotalCapture measured gyro 或 Custom measured gyro 上增加同类 extractor join control。

本 E3 没有运行模型训练或 pairing ablation。
