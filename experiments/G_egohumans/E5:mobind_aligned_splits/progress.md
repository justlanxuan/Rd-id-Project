# E5:mobind_aligned_splits 实时进度日志

## 2026-06-24 01:00
* A6 完成：生成对比表格、图表与 `results.md`。
* 核心结果（24 个 MoBInd official test 序列，双方均 unseen）：
  * **MoBInd (5s window)**: 0.9666
  * **Our pipeline (1 window)**: 0.9372
  * **Our pipeline (4-window vote)**: 0.9536
* 4-window vote 将差距从 2.94 pp 缩小到 1.30 pp。

## 2026-06-24 00:50
* A5 完成：MoBInd official checkpoint 在 aligned test set 上的 FrameAcc = 0.9666。
* 修复 motion batch reshape 问题，使 evaluation 可完整跑完 24 个序列。

## 2026-06-24 00:44
* A4 完成：our aligned model 4-window vote mean FrameAcc = 0.9536；1-window = 0.9372。

## 2026-06-24 00:30
* A2 完成：extract + slice 成功，123 个序列（排除 02_001–02_005，因为它们不在 MoBInd split 中）。
* 生成 windows：train 98 seq / val 6 seq / test 24 seq。
* 开始 A3：训练 aligned 4-IMU 模型。

## 2026-06-24 00:25
* Plan 已审批，创建 E5 沙盒。
* 完成 A1：撰写 `config/egohumans_mobind_aligned.yaml`。
* 开始 A2：运行 extract + slice。
