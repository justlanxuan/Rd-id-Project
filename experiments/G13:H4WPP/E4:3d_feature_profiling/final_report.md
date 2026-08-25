# G13 E4：H4W++ 3-D skeleton / 朝向特征 profiling 结果

## 结论

本实验完成 `14 features × 3 seeds × 4 LOSO folds = 168` 个独立训练/测试 run。
按预注册的三 seed 四折宏平均，最优特征是 `h36m3d_zonly`：只保留 H36M-17
root-relative、肩宽归一化后的深度 `z` 及其速度，得到 `80.55% ± 2.48%`。

它比当前 2-D `hybrid` baseline 的 `73.80% ± 0.41%` 高 `6.76` 个百分点，比完整
`h36m3d` 的 `72.79% ± 2.67%` 高 `7.77` 个百分点。z-only 在三个 seed 上分别为
`82.38% / 82.23% / 77.05%`，四个 held-out session 均保持相对稳定提升，但仍略低于
历史 E1 的 seed-42 单次 `80.89%`，因此不记录为新的 SOTA。

这个结果说明 Custom 当前配对任务中，H4W++ 的深度轴信息可能比完整的 2-D 关节形状
更有用；但由于四个 session 使用同一相机/采集域，z-only 也可能利用了相机深度或
session-specific bias。它应先做跨相机/跨域验证，再作为生产模型输入。

## 三 seed 四折宏平均

| feature | seed 0 | seed 42 | seed 123 | 3-seed mean ± population std | weighted mean |
|---|---:|---:|---:|---:|---:|
| **h36m3d_zonly** | 82.38% | 82.23% | 77.05% | **80.55% ± 2.48%** | **80.04%** |
| h36m3d_no_velocity | 72.93% | 73.11% | 75.52% | 73.85% ± 1.18% | 73.25% |
| hybrid (2-D baseline) | 74.30% | 73.79% | 73.29% | 73.80% ± 0.41% | 72.91% |
| h36m3d_right_wrist | 70.82% | 79.83% | 70.67% | 73.77% ± 4.28% | 72.91% |
| h36m3d_left_wrist_rot | 72.09% | 80.49% | 68.34% | 73.64% ± 5.08% | 72.85% |
| h36m2d | 74.76% | 75.08% | 70.22% | 73.35% ± 2.22% | 72.71% |
| h36m3d | 76.56% | 70.96% | 70.84% | 72.79% ± 2.67% | 72.10% |
| h36m3d_both_wrist | 71.60% | 75.49% | 67.89% | 71.66% ± 3.11% | 70.94% |
| h36m3d_left_wrist | 67.28% | 77.02% | 67.72% | 70.67% ± 4.49% | 69.79% |
| h36m3d_bone | 72.96% | 65.04% | 67.41% | 68.47% ± 3.32% | 67.62% |
| h36m3d_heading | 66.45% | 70.37% | 66.63% | 67.82% ± 1.81% | 67.07% |
| h36m3d_heading_rate | 69.89% | 71.57% | 61.02% | 67.49% ± 4.63% | 66.74% |
| h36m3d_geom | 55.96% | 74.66% | 67.34% | 65.99% ± 7.69% | 65.10% |
| h36m3d_rotinv | 59.37% | 63.50% | 70.94% | 64.60% ± 4.79% | 63.75% |

## 消融解释

- `h36m2d` 与 `h36m3d` 使用相同的 H36M-17 temporal tower，前者只将 z 置零；完整
  3-D position/velocity 没有带来收益，说明“加入全部 3-D 坐标”并不自动提升性能。
- `h36m3d_zonly` 的收益集中在深度轴，而不是 x/y 平面形状；这正是旧 2-D skeleton
  不具备的信息。
- `h36m3d_no_velocity` 与 baseline 接近，当前数据上速度通道不是主要增益来源。
- 显式 torso heading、heading rate 和 heading-invariant 版本均低于 baseline，说明
  当前坐标系下 heading 估计可能噪声较大或存在 session/camera frame 不一致。
- 左腕/右腕/双腕 proxy 都没有稳定超过 baseline。`left_wrist_rot` 的 seed 42 为
  `80.49%`，但 seed 123 只有 `68.34%`，不能视为可靠收益。
- 当前缓存只包含 H36M-17 body joints；腕部实验是左/右前臂方向及其 frame-to-frame
  rotation proxy，不是完整 MANO/SMPL-X wrist-local rotation。若要验证真正手腕旋转，
  需要下一阶段保留 H4W++ 的 hand joints 后重新抽取。

## Artifact

- 168 个 run：`/data/fzliang/reid-project/custom/artifacts/h4wpp_3d_feature_sweep/runs/`
- 汇总：同目录 `summary.json`、`seed_summary.csv`
- SHA-256：同目录 `artifact-manifest.json`
- 一键 runner：[tools/run_h4wpp_3d_feature_sweep.py](/home/fzliang/workspace/Re-id-Project/tools/run_h4wpp_3d_feature_sweep.py)
