# G9 Survey：已有数据与项目内证据

本文件先记录仓库内已有证据；外部文献调研不在本轮 Plan 批准范围内，后续若需要将作为独立 Survey 追加。

## 已有骨架源

- TotalCapture：Vicon/GT 作为 reference skeleton；已完成 G6 canonical preprocess。
- EgoHumans：已有 pose2d cache；S06 还保存了 AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer、WHAM 的统一 `.npz`。
- Custom：已有 AlphaPose 和 YOLO-Pose high 的原始/统一输出，并存在四个 held-out sessions。
- WHAM：原始输出包含 SMPL 参数、世界坐标和 mesh vertices，不等价于直接的 `skeleton.json`。

## 当前结果线索

1. G6 中 EgoHumans zero-shot 高于 fine-tune，提示需要先排查 target protocol、normalization、配对和过拟合。
2. TotalCapture zero-shot 与 fine-tune 差距较小，说明 source 表征和 target adaptation 的问题可能依赖 source。
3. G7 stride-24 变体对 EgoHumans 跨域结果影响较大，提示时间窗口/采样是候选 gap。
4. G8 的 tracklet history 在不同 session/seed 间不稳定，提示身份关联和历史状态不能与表征质量混为一谈。

这些都是待验证假设，不是最终因果结论。

## 数据资产入口

- G6 正式结果：`/data/fzliang/reid-project/g6/c9a5d3099979296a72314eba66274855e03ab1eb/`
- EgoHumans 多算法结果：`/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/Experiment/RB-Skeleton-Aug/S06_Algo_Aug/`
- Skeleton extractor 代码和环境：`/data/lyxie/ReID/Pipeline/Skeleton_Extractors/`
- Custom YOLO-Pose high：`/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/data/custom_annotation_video_pose/yolo_pose_high/`
- Custom/TotalCapture/EgoHumans G6 canonical artifacts：`/data/fzliang/reid-project/`
