# E4:single_imu_right_wrist 实时进度日志

## 2026-06-23 23:53
* A3/A4/A5 全部完成。
* 核心结果（16 个 train-only 序列）：
  * 4-IMU 1-window FrameAcc = **0.9562**
  * 1-IMU 1-window FrameAcc = **0.8080**（↓ 14.81 pp）
  * 4-IMU 4-window vote FrameAcc = **0.9632**
  * 1-IMU 4-window vote FrameAcc = **0.8653**（↓ 9.80 pp）
* Full test（20 sequences）：4-IMU 0.9558 vs 1-IMU 0.7934（↓ 16.25 pp）。
* 已生成 `results/results.md`、对比图、`comparison_summary.json`。

## 2026-06-23 23:52
* A2 训练完成（11m 22s）。
* Best val top1 = 0.7064（epoch 39），最终 epoch 50 val top1 = 0.7022。
* Checkpoint: `data/interim/egohumans_full_extract/train/egohumans_right_wrist/best.pt`
* 开始 A3/A4：在 full test 与 16 train-only 子集上评估。

## 2026-06-23 22:20
* Plan 已审批，创建 E4 沙盒。
* 完成 A1：撰写 `config/egohumans_right_wrist.yaml`。
* 开始 A2：启动 1-IMU 模型训练。
