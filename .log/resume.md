# 🔄 HAROS Task Resume Node

> ⚠️ **写给 AI:** 本文件是人机协作的唯一状态机断点。在每次迭代结束或实验中断前，必须精确更新以下字段，严禁留空。

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E3:mobind_vs_pipeline_frameacc 已完成**
* **当前具体执行任务:** 在完全相同的 16 个 MoBInd-train-only EgoHumans 序列上，对齐 MoBInd 官方 checkpoint 与我们 trained pipeline 的 FrameAcc 指标。
* **当前子任务:** E3 结果已汇总，等待最终 commit/push。

## 2. 最新成果
* E3 实验沙盒已按 HAROS 规范创建并完结：
  * `experiments/G_egohumans/E3:mobind_vs_pipeline_frameacc/plan.md`
  * `experiments/G_egohumans/E3:mobind_vs_pipeline_frameacc/progress.md`
  * `experiments/G_egohumans/E3:mobind_vs_pipeline_frameacc/test/test.md`
  * `scripts/A1_eval_mobind_frameacc.py`
  * `scripts/A2_eval_ours_frameacc_subset.py`
  * `scripts/A4_eval_ours_frameacc_4window.py`
  * `scripts/A3_compare_results.py`
  * `results/results.md`
  * `results/figures/frameacc_comparison.png`
* 核心结果（16 个 train-only 序列）：
  * **MoBInd official stage2**: mean FrameAcc = **0.9654**
  * **Our pipeline (1 window)**: mean FrameAcc = **0.9562**
  * **Our pipeline (4 windows aggregated)**: mean FrameAcc = **0.9576**
  * 差距：MoBInd 领先 1-window **+0.93 pp**，领先 4-window **+0.78 pp**
* E2 基线仍保留：Retrieval R@1 ~83%，Localization Person 98.0% / Limb 89.2%，Sync Acc@0.2 ~99–100%。
* MoBind 本地修改已保存为 patch：`third-party/mobind_egohumans_fixes.patch`。

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* 需注意：E3 使用了与 E2 相同的 MoBInd 官方 checkpoint；所有对比序列均来自 MoBInd train split，无 test-set 泄漏。

## 4. 下一步行动 (Next Actions)
* [ ] 将 E3 实验区文件强制 add 并 push 到 `egohumans` 分支。
* [ ] 基于 E3 结论，考虑后续改进实验（E4），例如：
  * 在同样 16 序列上用 MoBInd 的 5 秒窗口重新训练/微调我们的模型；
  * 将 MoBInd 的 IMU encoder 作为我们 pipeline 的初始化；
  * 消融不同输入预处理（COCO pose2d vs H36M skeleton）对 FrameAcc 的影响。
