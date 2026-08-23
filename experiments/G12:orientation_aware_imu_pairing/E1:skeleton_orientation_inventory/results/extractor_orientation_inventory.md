# E1′：我们自己的骨架提取器输出朝向审计

## 范围修正

本表审计的是 `/data/lyxie` 下 Re-ID 项目实际运行的提取器及其进入配对输入的输出，不是数据集自带的 Vicon/SMPL 朝向。此前的 `orientation_inventory.md` 保留为 source-orientation reference；本表是 G12 的主 E1 入口。

审计的 canonical 输入为：

- `/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/Experiment/RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs/<method>/*.npz`
- `/data/lyxie/ReID/Pipeline/Re-id-Project-egohumans/data/skeleton_aug/S06_source_ablation/<method>/.../sequences/*.npz`

两者均使用 `skeleton`/`extract_skeleton` 的 `(T,N,17,3)` 表示；S06 metadata 明确写为 `root-centered torso-scaled H36M17 xyz`。YOLO-Pose high 与 AlphaPose 的第三维为 0（2D 来源的 3D-compatible padding），不是深度或世界坐标。

## 结果

| 提取器 | 实际 canonical 样本 | canonical 是否含 rotation/root 字段 | 朝向级别 | 可否直接与 IMU gyro 配对 |
|---|---:|---|---|---|
| YOLO-Pose high | 88 algorithm / 304 S06 | 否；全量 canonical scan 均为 `skeleton (T,N,17,3)`，z=0 | 2D image-plane proxy | 否。只能从肩/胯关节构造视角依赖的弱 proxy |
| AlphaPose | 88 / 304 | 否；全量 COCO17→H36M17 输出仍只有关节位置、visibility | 2D image-plane proxy | 否。不能当世界 yaw；还受检测/跟踪遮挡影响 |
| FMPose3D | 88 / 304 | 否；全量 metadata 为 `root-centered torso-scaled H36M17 xyz` | 3D joints-derived heading | 可以做坐标系依赖的 torso heading/变化率，但不是直接 root rotation |
| MotionAGFormer | 88 / 304 | 否；同上 | 3D joints-derived heading | 可以做 3D 几何 heading；需检查 lifting 的相机/屏幕坐标语义 |
| TCPFormer | 88 / 304 | 否；同上 | 3D joints-derived heading | 可以做 3D 几何 heading；必须保留有效性和退化标记 |
| WHAM | 88 / 304 | 否；canonical adapter 全量只保留 17 joints | `direct_orientation_raw_but_not_canonical` | raw WHAM 可用直接 root orientation；当前配对输入没有传递它 |

### WHAM 的关键边界

WHAM 原始处理产物仍保留直接朝向字段：

- `/data/lyxie/ReID_imu_generation/outputs/wham/imu/processed_smoke/..._smplx.npz` 含 `root_orient (T,3)` axis-angle、`pose_body`、`poses`；
- `/data/lyxie/ReID_imu_generation/outputs/wham/recon/.../processed/*.npz` 含 `pose`、`pose_world`、`trans`、`trans_world`，metadata 标注 camera/local 或 world 语义。

但 S06 的 `algorithm_outputs/wham/*.npz` 与最终 `extract_skeleton` 只含 `skeleton`、`visibility`、frame/person identity 和 alignment metadata，没有 `root_orient`/`pose_world`。因此“WHAM 能估计朝向”不能等同于“当前模型已经看到了 WHAM 朝向”；需要新增字段传递和坐标契约后才能进入 E2/E4。

## 结论

1. 当前六类 extractor 的 canonical 配对输入中，没有一个把直接 root/global rotation 传到模型；WHAM 是唯一已在 raw artifact 中观测到直接 root orientation、但在 adapter 处丢失的例外。
2. FMPose3D、MotionAGFormer、TCPFormer 和 canonical WHAM 的 3D 关节仍包含相对身体几何，可推导 torso heading；这属于 `derived`，不能冒充世界朝向。
3. YOLO-Pose high、AlphaPose 只提供 2D image-plane proxy。其肩线/髋线方向可作为 diagnostic，但没有单目深度、相机姿态或全局 yaw 可识别性。
4. 下一步应先做 extractor-level E2：统一关节索引、root/torso heading 定义、3D 坐标系、退化/遮挡 mask、真实 timestamp derivative；同时为 WHAM 做 raw `root_orient` 传递的独立 contract，不把它与 3D-derived heading 混表。

## 可复现证据

- 生成脚本：[E1_build_extractor_orientation_inventory.py](../scripts/E1_build_extractor_orientation_inventory.py)
- JSON artifact：`/data/fzliang/reid-project/g12/e1_inventory/extractor_orientation_inventory.json`
- 运行命令：

  ```bash
  /data/lyxie/ReID/Pipeline/Skeleton_Extractors/2D/AlphaPose/venv/bin/python \
    experiments/G12:orientation_aware_imu_pairing/E1:skeleton_orientation_inventory/scripts/E1_build_extractor_orientation_inventory.py \
    --output /data/fzliang/reid-project/g12/e1_inventory/extractor_orientation_inventory.json
  ```

- 记录了每个 method 的 canonical/algorithm/raw 文件计数、sample keys/shapes/dtype/finite 范围、orientation-like keys 和 sample SHA256；未用空结果替代读取失败。
- 全量 304 canonical + 88 algorithm 文件逐一检查：六种方法均 `errors=0`、`nonfinite_skeleton_files=0`、`orientation_key_files=0`；raw WHAM 另以轻量计数/样本字段审计，避免把 6890-vertex mesh 全部载入内存。
