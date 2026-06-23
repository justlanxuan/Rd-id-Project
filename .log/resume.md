# 🔄 HAROS Task Resume Node

> ⚠️ **写给 AI:** 本文件是人机协作的唯一状态机断点。在每次迭代结束或实验中断前，必须精确更新以下字段，严禁留空。

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E4:single_imu_right_wrist 已完成**
* **当前具体执行任务:** 在完全相同的 EgoHumans 数据划分上，对比 4-IMU pipeline 与仅使用右手腕 IMU（R_LowArm）pipeline 的同步匹配性能。
* **当前子任务:** E4 结果已汇总，等待最终 commit/push。

## 2. 最新成果
* E4 实验沙盒已按 HAROS 规范创建并完结：
  * `experiments/G_egohumans/E4:single_imu_right_wrist/plan.md`
  * `experiments/G_egohumans/E4:single_imu_right_wrist/progress.md`
  * `experiments/G_egohumans/E4:single_imu_right_wrist/test/test.md`
  * `config/egohumans_right_wrist.yaml`
  * `scripts/A3_eval_1imu_full_test.py`
  * `scripts/A4_eval_1imu_subset.py`
  * `scripts/A5_compare_4imu_vs_1imu.py`
  * `results/results.md`
  * `results/figures/frameacc_4imu_vs_1imu.png`
  * `results/comparison_summary.json`
* 核心结果（16 个 MoBInd-train-only 序列）：
  * **4-IMU 1-window**: mean FrameAcc = **0.9562**
  * **1-IMU 1-window**: mean FrameAcc = **0.8080**（↓ 14.81 pp）
  * **4-IMU 4-window vote**: mean FrameAcc = **0.9632**
  * **1-IMU 4-window vote**: mean FrameAcc = **0.8653**（↓ 9.80 pp）
* Full test（20 sequences）：4-IMU 0.9558 vs 1-IMU 0.7934（↓ 16.25 pp）。
* 1-IMU 模型 checkpoint：`data/interim/egohumans_full_extract/train/egohumans_right_wrist/best.pt`

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* 结果提示：仅使用右手腕 IMU 会导致 FrameAcc 显著下降，说明多传感器信息对当前 IMU-to-person 匹配任务仍然关键。

## 4. 下一步行动 (Next Actions)
* [ ] 将 E4 实验区文件强制 add 并 push 到 `egohumans` 分支。
* [ ] 基于 E4 结论，可考虑后续实验：
  * 测试 2-IMU 组合（如双手腕 R_LowArm + L_LowArm）以寻找传感器数量与性能的平衡点；
  * 探索 wrist + ankle 的混合组合；
  * 研究 1-IMU 失败序列（如 `01_011`、`02_001`）的特征，看是否可通过时序聚合或数据增强弥补。
