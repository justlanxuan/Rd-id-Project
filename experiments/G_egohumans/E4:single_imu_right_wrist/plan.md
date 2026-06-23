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

## 关键控制变量
- 数据：复用 `data/interim/egohumans_full_extract/slice/` 下的 NPZ 与 CSV。
- 仅改变 `train.imu_sensor`（空 → `R_LowArm`）与 `train.repeat_single_sensor`（1 → 4）。
- 输出目录隔离：通过 `train/test.output.run_name` 区分，不覆盖 4-IMU checkpoint。
