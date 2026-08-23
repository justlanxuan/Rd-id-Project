# G12 Survey：现有骨架朝向数据与可用性

审计日期：2026-08-21。以下结论来自本机数据文件的只读 schema 检查；不把文件存在等同于坐标语义已经验证。

## 1. 直接 orientation 来源

### TotalCapture Vicon

`/data/lyxie/ReID/Data/totalcapture/s*_vicon_pos_ori.tar.gz` 包含每个序列的 `gt_skel_gbl_ori.txt`。已抽查的 `/data/lyxie/ReID_imu_generation/data/raw/totalcapture/S1_freestyle3/gt_skel_gbl_ori.txt` 有 21 个关节、每关节 4 个数值，四元数范数约为 1；`Hips` 是最直接的 body/root orientation 候选。

对应的 `gt_skel_gbl_pos.txt` 提供 21 个关节的全局 3D 位置。当前项目的 `preprocess/datasets/totalcapture.py` 只解析 position，尚未把 `gt_skel_gbl_ori.txt` 传入 canonical NPZ。

### TotalCapture SMPL-X

`/data/lyxie/ReID_imu_generation/data/processed/totalcapture_test/*/*_smplx.npz` 包含 `root_orient (T,3)`、`pose_body`、`trans` 等字段。`root_orient` 是轴角形式的根部全局旋转，需在使用前确认与 Vicon quaternion 的坐标约定一致。

### EgoHumans fitted SMPL

`/data/lyxie/ReID/Data/egohumans/data/*/processed_data/smpl/*.npy` 的每帧记录含 `global_orient`、`body_pose`、`transl`、`vertices` 和 `joints`；当前扫描约 70,113 个 SMPL 文件。它是直接的 SMPL 全局朝向字段，但属于拟合/估计产物，需要单独记录估计误差、相机/世界坐标和 frame provenance。

## 2. 可推导但非直接的 orientation 来源

- EgoHumans `extracted_data/*.npy` 的 `pose3d (T,24,3)` 可由 pelvis/shoulder/hip 几何推导 torso heading，但不能直接称为 yaw label。
- `/data/fzliang/reid-project/totalcapture/preprocessed/...` 的 `gt_skeleton_meters (T,1,17,3)` 保留了 3D 关节位置，因此理论上可推导 heading，但原始 Vicon orientation 已在 canonical adapter 中丢失。
- EgoHumans full `fit_poses3d/refine_poses3d` 是 3D 关节结果，适合做 derived-heading 对照；其相机/世界坐标语义需在 E1 中确认。

## 3. 当前没有可靠世界朝向的来源

- `/data/fzliang/reid-project/egohumans/preprocessed/...` 当前 canonical `gt_skeleton` 是 2D xy 加 visibility。
- `/data/fzliang/reid-project/custom/preprocessed/g11_complete_w24_stride12_v1` 的 `skeleton`/`extract_skeleton` 是 2D skeleton；没有 root quaternion、SMPL global orientation 或世界 yaw。
- AlphaPose/COCO17 detector 输出只有 2D joints、confidence 和 visibility。
- `/data/fzliang/data/reconstruct` 是 IMU trajectory/position/velocity/acceleration，不是人物 skeleton orientation。

这些信号可用于 orientation-missing、2D torso proxy 或 source-only control，但不能当成 Custom 世界朝向真值。

## 4. 当前代码缺口

正式 preprocess adapter 只保存 position/skeleton 和 IMU。仓库中出现的 `rotation` 主要属于 IMU quaternion/rotation-matrix 转换，不是人物 root orientation。G12 首轮不直接修改公共 schema，而先完成来源、坐标、时间和推导契约审计。

## 5. 结论

最可信的首轮顺序是：TotalCapture direct orientation → EgoHumans fitted SMPL orientation → 3D-joint-derived heading → 2D proxy/Custom missingness。只有前三类的语义和时间对齐通过审计后，才进入 orientation-aware pairing ablation。
