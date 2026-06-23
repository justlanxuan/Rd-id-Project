# 🔄 HAROS Task Resume Node

> ⚠️ **写给 AI:** 本文件是人机协作的唯一状态机断点。在每次迭代结束或实验中断前，必须精确更新以下字段，严禁留空。

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E3b:unseen_sequences_correction 已完成**
* **当前具体执行任务:** 修正 E3 中测试序列选择的问题，确保 MoBInd 官方 checkpoint 与我们的 4-IMU pipeline 在双方都 unseen 的序列上对比 FrameAcc。
* **当前子任务:** E3b 结果已汇总，等待最终 commit/push。

## 2. 最新成果
* 已识别 E3 原 16-sequence "train-only" subset 的问题：那些序列属于 MoBInd 官方 train split，MoBInd checkpoint 在训练时见过它们。
* 新的严格 unseen 子集：我们 test_sessions 与 MoBInd test/val split 的交集，共 4 个序列：
  ```
  01_002, 03_001, 04_005, 05_002
  ```
* E3b 实验沙盒已按 HAROS 规范创建：
  * `experiments/G_egohumans/E3b:unseen_sequences_correction/plan.md`
  * `experiments/G_egohumans/E3b:unseen_sequences_correction/progress.md`
  * `experiments/G_egohumans/E3b:unseen_sequences_correction/scripts/B1_eval_unseen_sequences.py`
  * `experiments/G_egohumans/E3b:unseen_sequences_correction/scripts/B2_compare_unseen.py`
  * `experiments/G_egohumans/E3b:unseen_sequences_correction/results/results.md`
  * `experiments/G_egohumans/E3b:unseen_sequences_correction/results/figures/frameacc_unseen_comparison.png`
* 修正后的 FrameAcc（4 个 unseen 序列）：
  * **MoBInd (5s window)**: **0.9717**
  * **Our pipeline (1 window)**: **0.9546**（落后 1.71 pp）
  * **Our pipeline (4-window vote)**: **0.9717**（持平）
* E4 单 IMU 结果仍保留：1-IMU 在 full test 上 FrameAcc 0.7934，显著低于 4-IMU。

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* 需注意：unseen subset 仅 4 个序列，样本量小，结论需谨慎推广。

## 4. 下一步行动 (Next Actions)
* [ ] 将 E3b 实验区文件强制 add 并 push 到 `egohumans` 分支。
* [ ] 在论文/报告中使用修正后的 E3b 数据，并明确说明原 E3 16-sequence subset 的偏差。
* [ ] 可考虑在更大的 MoBInd test/val 集合上扩展验证，或直接使用 full 20-sequence test set 作为补充参考。
