# E4:single_imu_right_wrist 实时进度日志

## 2026-06-27 09:15
* 在人类授权下启动 E4 子实验 A6：分析不同 test clip 在 4-IMU -> 1-IMU 后准确率下降幅度的规律。
* A6b 分析脚本已完成：计算了活动丰富度、右腕 IMU 主导性、骨架右腕运动占比、静态帧比例、姿态多样性、4-IMU 基线准确率等指标。
* 关键发现：
  - **tagging** 类 clip 下降最严重（平均 Δ ≈ 46.9 pp），**tennis** 下降最小（平均 Δ ≈ 4.0 pp）。
  - 4-IMU 基线准确率与下降幅度呈负相关（Spearman r=-0.37）：基线越难的 clip，减少传感器后崩得越厉害。
  - 姿态多样性与下降幅度正相关（Spearman r=0.36），提示复杂姿态更依赖全身多传感器信息。
* 产物：
  - `results/A6_activity_richness_analysis/results.md`
  - `results/A6_activity_richness_analysis/activity_richness_metrics.{json,csv}`
  - `results/A6_activity_richness_analysis/delta_vs_*.png`、`delta_per_clip.png`、`delta_per_activity.png`
* A6a 视频合成已完成：20 个 test clip 全部合成为 `vis/test_clips/*.mp4`（共 1.5 GB），可直接播放查看。
* 人类补充假设：下降幅度可能与同视频人物个数、跨人动作差异、跨人手腕运动差异有关。
* A6b 已扩展：新增 `num_persons`、`wrist_sync_abs_mean`、跨人姿态距离/时序相关、跨人运动同步、跨人四肢能量距离等指标。
* 验证结论：
  - 人数、跨人能量/速度差异与 `delta_pp` 的相关性均较弱（|Spearman r| < 0.22，p > 0.37）。
  - 但 **跨人姿态时序相关性**（`mean_pose_temporal_corr`）与 `delta_pp` 呈正相关趋势（Spearman r=0.31, p=0.19；Pearson r=0.44, p=0.055）：人物姿态演变越同步/相似，单 IMU 下降越明显，与“tagging 大家做类似动作”的直觉一致。
  - 说明“人与人之间活动相似度”需要用**姿态/动作轨迹的时序相似性**来刻画，而非简单的能量差异。
* A6 全部子任务完成。

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
