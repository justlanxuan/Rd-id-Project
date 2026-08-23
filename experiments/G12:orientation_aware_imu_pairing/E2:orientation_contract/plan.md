# E2 Plan：Extractor orientation contract

## 目标

在不修改正式 `extract_skeleton` schema 的前提下，冻结三条可复核的 orientation 轨道：

1. `2d_proxy`：YOLO-Pose high/AlphaPose 的 image-plane shoulder axis；
2. `3d_derived`：FMPose3D/MotionAGFormer/TCPFormer/WHAM canonical 3D joints 推导 torso heading；
3. `direct`：WHAM raw `root_orient`/`pose_world` 等 axis-angle 或 quaternion。

另设 `orientation_missing`，禁止以零值、2D proxy 或文件名推断世界 yaw。

## 输入与坐标契约

- 统一输入为 `[T,J,C]`，关节名必须随 artifact 传入；默认 H36M17 顺序见 `src/features/global_motion.py`。
- `2d_proxy` 只接受 `C=2`，肩轴为 `right_shoulder - left_shoulder`，角度周期为 π，前后方向不可辨识。
- `3d_derived` 只接受 `C=3`，`lateral = right_shoulder-left_shoulder`，`up = thorax-pelvis`，`forward = cross(lateral, up)` 或显式反向；up axis、cross order 和 coordinate frame 必须写入 manifest。
- `direct` 的 axis-angle 使用弧度；quaternion 顺序必须声明为 `wxyz` 或 `xyzw`；local forward axis/sign、world up axis 和 frame 必须声明。
- 6D 表示为 rotation matrix 的前两列，按行展开为 6 个值。

## 时间、有效性和缺失

- timestamp 必须是严格递增秒值；禁止用 frame index 或文件名猜采样率。
- angle rate 在连续 valid segment 内按真实 timestamp 求导；segment 断点不跨越 missing/degenerate frame。
- `orientation_valid` 表示角度/方向有效，`rate_valid` 表示导数有效；退化原因必须保留。
- 2D line angle 输出 `[sin(2θ), cos(2θ)]`，3D/direct 输出 `[sin(θ), cos(θ)]`。

## Gate

- quaternion normalization/sign continuity、axis-angle conversion、angle wrap、irregular timestamp derivative、2D π-periodicity、3D cross-order flip、degeneracy/missingness tests 全部通过；
- contract 只生成独立 feature object，不回写 G10/G11 或 canonical NPZ；
- E2 通过前不运行 orientation-aware pairing ablation。
