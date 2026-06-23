# E3:mobind_vs_pipeline_frameacc 实时进度日志

## 2026-06-23 18:30
* 新增 A5：4 窗口 assignment 投票聚合。
* 最终结果（16 个 train-only 序列）：
  * MoBInd mean FrameAcc = **0.9654**
  * Our pipeline (1 window) = **0.9562**
  * Our pipeline (4 windows mean) = **0.9576**
  * **Our pipeline (4 windows vote) = 0.9632**
* 投票聚合将差距缩小到仅 **0.22 pp**。
* 更新 `results/results.md`、图表、HAROS 文档。

## 2026-06-23 18:00
* 新增 A4：我们的 pipeline 使用 4 窗口 embedding 平均后重新评估 FrameAcc。
* 结果更新：MoBInd 0.9654 / Ours(1w) 0.9562 / Ours(4w-mean) 0.9576。

## 2026-06-23 17:30
* 完成 A1/A2/A3 全部实验。
* MoBInd mean FrameAcc = 0.9654；Our pipeline = 0.9562。
* 生成对比图表与 `results/results.md`。

## 2026-06-23 17:00
* Plan 已审批通过，创建 E3 实验沙盒。
