# E5 Results: MoBInd-Aligned Splits on MoBInd Official Test Set

## 说明

本实验使用 **MoBInd 官方 train/val/test action split** 重新训练我们的 4-IMU pipeline，确保在测试的 24 个序列上，**MoBInd 官方 checkpoint 与我们的模型都未见过这些序列**。

## FrameAcc 对比（24 个 MoBInd test 序列）

| Sequence | MoBInd (5s window) | Our pipeline (1 window) | Our pipeline (4-window vote) |
|---|---|---|---|
| custom_01_001 | 0.9704 | 0.8648 | 0.8884 |
| custom_01_002 | 0.9789 | 0.8971 | 0.9406 |
| custom_01_003 | 0.9551 | 0.7113 | 0.7699 |
| custom_01_004 | 0.9264 | 0.8788 | 0.9264 |
| custom_03_001 | 0.9804 | 0.9804 | 0.9804 |
| custom_03_002 | 0.9928 | 0.9928 | 0.9928 |
| custom_03_003 | 0.9731 | 0.9731 | 0.9731 |
| custom_03_004 | 0.9817 | 0.9817 | 0.9817 |
| custom_04_001 | 0.9844 | 0.9740 | 0.9844 |
| custom_04_002 | 0.9303 | 0.9346 | 0.9373 |
| custom_04_003 | 0.9557 | 0.9289 | 0.9631 |
| custom_04_004 | 0.9094 | 0.9094 | 0.9094 |
| custom_05_001 | 0.9816 | 0.9521 | 0.9700 |
| custom_05_002 | 0.9660 | 0.8792 | 0.9660 |
| custom_05_003 | 0.9808 | 0.9467 | 0.9808 |
| custom_05_004 | 0.9601 | 0.9260 | 0.9501 |
| custom_06_001 | 0.9362 | 0.9362 | 0.9362 |
| custom_06_002 | 0.9062 | 0.9045 | 0.9062 |
| custom_06_003 | 0.9682 | 0.9682 | 0.9682 |
| custom_06_004 | 0.9701 | 0.9701 | 0.9701 |
| custom_07_001 | 0.9992 | 0.9992 | 0.9992 |
| custom_07_002 | 0.9992 | 0.9992 | 0.9992 |
| custom_07_003 | 1.0000 | 1.0000 | 1.0000 |
| custom_07_004 | 0.9937 | 0.9837 | 0.9937 |
| **Mean** | **0.9666** | **0.9372** | **0.9536** |

## 结论

- 在严格对齐的 MoBInd test set 上，MoBInd 的 mean FrameAcc 为 **0.9666**。
- 我们的 pipeline 1-window 为 **0.9372**（落后 2.95 pp）。
- 我们的 pipeline 4-window vote 为 **0.9536**（落后 1.30 pp）。

## 与之前实验的关系

- E3 原 16-sequence subset 因使用 MoBInd train split 序列而不够公平。
- E3b 的 4-sequence unseen subset 结论与本次 24-sequence 结果一致：4-window vote 能大幅缩小与 MoBInd 的差距。
- 本次 E5 是样本量最大、最严格的公平对比。

## AI Reflection

- 数据划分对齐后，我们的 pipeline 仍略低于 MoBInd，但差距在 1 pp 左右。
- 4-window vote 再次证明是有效的决策级聚合策略。
- 后续可尝试使用 MoBInd 的 5 秒窗口或将其 IMU encoder 作为初始化，进一步缩小差距。
