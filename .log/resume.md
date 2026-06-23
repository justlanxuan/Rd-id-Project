# 🔄 HAROS Task Resume Node

> ⚠️ **写给 AI:** 本文件是人机协作的唯一状态机断点。在每次迭代结束或实验中断前，必须精确更新以下字段，严禁留空。

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E5:mobind_aligned_splits 已完成**
* **当前具体执行任务:** 使用与 MoBInd 官方 action-split 对齐的 train/val/test 重新训练我们的 4-IMU pipeline，并在 24 个 MoBInd official test 序列上与 MoBInd 官方 checkpoint 做严格公平对比。
* **当前子任务:** E5 训练、评估与结果汇总均已完成，等待 commit/push。

## 2. 最新成果
* 已识别旧划分的偏差：我们的旧 train 包含 15 个 MoBInd test 序列，旧 test 包含 15 个 MoBInd train 序列，导致 E3/E4 无法直接与 MoBInd 公平对比。
* 采用 MoBInd 官方 action split 生成新 `windows_*.csv`：train 98 seq / val 6 seq / test 24 seq。
* 使用相同超参数重新训练 4-IMU 模型（50 epochs，frozen backbone）并保存到 `data/interim/egohumans_mobind_aligned/train/egohumans_mobind_aligned/best.pt`。
* 在 24 个 unseen test 序列上的 FrameAcc：
  * **MoBInd official (5s window)**: **0.9666**
  * **Our pipeline aligned (1 window)**: **0.9372**（落后 2.94 pp）
  * **Our pipeline aligned (4-window vote)**: **0.9536**（落后 1.30 pp）
* E5 实验沙盒已按 HAROS 规范创建并生成最终结果：
  * `experiments/G_egohumans/E5:mobind_aligned_splits/results/results.md`
  * `experiments/G_egohumans/E5:mobind_aligned_splits/results/figures/frameacc_mobind_aligned.png`
  * 评估脚本 `A4_eval_ours_aligned_test.py`、`A5_eval_mobind_aligned_test.py`、`A6_compare_aligned.py`

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* 注意：E5 结果略低于 MoBInd，但差距缩小到 1 pp 级别，说明数据划分对齐是关键。

## 4. 下一步行动 (Next Actions)
* [ ] 将 E5 实验区文件强制 add 并 push 到 `egohumans` 分支。
* [ ] 在论文/报告中使用 E5 作为主要的 MoBInd 公平对比结果，并明确说明 E3/E4 的划分偏差。
* [ ] 后续可尝试使用 MoBInd 的 5 秒窗口或将其 IMU encoder 作为初始化，进一步缩小差距。
