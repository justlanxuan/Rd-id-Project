# E4:single_imu_right_wrist 实验计划

## 目标
训练并评估仅使用 **右手腕 IMU (`R_LowArm`)** 的 EgoHumans pipeline，与已有 4-IMU 模型进行严格对比，重点监测逐帧 FrameAcc。

## 对照组
- 4-IMU 模型：`configs/egohumans_full_extract.yaml` → `data/interim/egohumans_full_extract/train/egohumans_full_extract/best.pt`
- 已有结果（E3）：full test FrameAcc 0.9558；16 train-only sequences 1-window 0.9562 / 4-window vote 0.9632。

## 实验组
- 1-IMU 模型：仅使用 `R_LowArm`，通过 `repeat_single_sensor: 4` 保持 48-D 输入。
- 其他训练参数（backbone、epochs、batch size、冻结策略、窗口、数据）全部与 4-IMU 保持一致。

## 子实验

| 编号 | 内容 | 产物 |
|---|---|---|
| A1 | 准备专用配置 `config/egohumans_right_wrist.yaml` | 配置文件 |
| A2 | 训练 1-IMU 模型 | `data/interim/egohumans_full_extract/train/egohumans_right_wrist/best.pt` |
| A3 | 在完整 20-sequence 测试集上评估 | `results/full_test_1imu.json` |
| A4 | 在 16 train-only 序列上评估（1-window + 4-window vote） | `results/ours_1imu_1window.json`, `results/ours_1imu_4window_vote.json` |
| A5 | 与 4-IMU 对比并可视化 | `results/results.md`, `results/figures/frameacc_4imu_vs_1imu.png` |
| **A6** | **Clip 级下降规律诊断：活动丰富度、静态帧比例与可视化** | `vis/test_clips/*.mp4`, `results/A6_activity_richness_analysis/` ✅ 已完成 |

## A6 子实验细节
- **A6a**：将 20 个 test clip 的 `cam03` 图像帧合成为 mp4，保存到 `vis/test_clips/`。
- **A6b**：从 MoBInd NPZ 计算每个 clip 的活动丰富度指标：
  - 右腕 IMU 运动能量（`wrist_energy_mean/std`）
  - 全 IMU 运动能量（`full_imu_energy_mean/std`）
  - 全身骨架根节点速度（`skeleton_velocity_mean/std`）
  - 姿态多样性（`pose_diversity` = 各关节时序标准差均值）
  - 静态帧比例（`static_frame_ratio`，基于 wrist energy 阈值）
  - 时长、人数、活动类别
- **A6c**：计算每个 clip 从 4-IMU 降到 1-IMU 的 FrameAcc 下降幅度 `Δ = Acc_4imu - Acc_1imu`。
- **A6d**：做相关性分析与可视化（散点图 + Spearman/Pearson），撰写 `results/A6_activity_richness_analysis/results.md`，解释“为什么有些 clip 下降远高于其他”。

## 关键控制变量
- 数据：复用 `data/interim/egohumans_full_extract/slice/` 下的 NPZ 与 CSV。
- 仅改变 `train.imu_sensor`（空 → `R_LowArm`）与 `train.repeat_single_sensor`（1 → 4）。
- 输出目录隔离：通过 `train/test.output.run_name` 区分，不覆盖 4-IMU checkpoint。
