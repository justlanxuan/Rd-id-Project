# E3b Corrected Results: Unseen-by-Both Sequences

## 说明

E3 之前使用的 16 个序列属于 MoBInd 官方 train split，导致 MoBInd checkpoint 在训练时见过它们，不是严格的泛化测试。本实验改用 **双方都没见过的 4 个序列**（我们的 test_sessions 与 MoBInd test/val 的交集）：

```
01_002, 03_001, 04_005, 05_002
```

## FrameAcc 对比

| Sequence | MoBInd (5s window) | Our pipeline (1 window) | Our pipeline (4-window vote) |
|---|---|---|---|
| custom_01_002 | 0.9789 | 0.9393 | 0.9789 |
| custom_03_001 | 0.9804 | 0.9804 | 0.9804 |
| custom_04_005 | 0.9617 | 0.9494 | 0.9617 |
| custom_05_002 | 0.9660 | 0.9492 | 0.9660 |
| **Mean** | **0.9717** | **0.9546** | **0.9717** |

## 结论

- 在严格 unseen 的 4 个序列上，MoBInd 的 mean FrameAcc 为 **0.9717**。
- 我们的 pipeline 1-window 为 **0.9546**（落后 1.72 pp）。
- 我们的 pipeline 4-window vote 为 **0.9717**，与 MoBInd 基本持平（差距 0.00 pp）。

## AI Reflection

- 之前 E3 的 16-sequence 'train-only' subset 对 MoBInd 不够公平；本 4-sequence unseen subset 才是严格对齐的对比。
- 4-window vote 在这个小集合上表现出色，与 MoBInd 5s 窗口相当，但样本量仅 4 个序列，结论需谨慎推广。
- 建议后续在更大的 unseen 集合上验证，或直接使用完整 20-sequence test set 作为 secondary 参考。
