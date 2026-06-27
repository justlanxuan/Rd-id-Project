# E4:single_imu_right_wrist 测试说明

## 测试目标
验证仅使用右手腕 IMU 的模型能够完成训练，并在与 4-IMU 模型相同的数据划分上输出可对比的同步匹配指标。

## 测试项

| 子实验 | 对象 | 通过标准 |
|---|---|---|
| A1 | `config/egohumans_right_wrist.yaml` | YAML 可正常解析，`imu_sensor` 为 `R_LowArm`，`repeat_single_sensor` 为 4 |
| A2 | 训练脚本 | 正常跑完 50 epochs，生成 `best.pt` 与 `imu_stats.json`，无 OOM |
| A3 | pipeline test stage | 生成 full test 的 synchronous/grouped 结果，包含 FrameAcc |
| A4 | `src/engine/eval_synchronous.py` 在 16-train-only CSV 上 | 输出 1-window 与 4-window vote 的 FrameAcc |
| A5 | 对比脚本 | 生成 `results/results.md` 与对比图 |
| A6 | Clip 级下降规律诊断 | 生成 `results/A6_activity_richness_analysis/results.md` 与 `vis/test_clips/*.mp4` |

## 失败判定
- 训练报错 `Unsupported sensor_name=R_LowArm` → 检查 sensor 名称拼写与 `alignment_dataset.py` 中的 `order` 列表。
- 训练输出覆盖了 4-IMU checkpoint → 检查 `run_name` 与 `output_root`。
- 评估结果中 1-IMU 的 FrameAcc 异常低（<0.7）且伴随 NaN → 检查 IMU stats 是否正确计算（`compute_imu_stats: true`）。
