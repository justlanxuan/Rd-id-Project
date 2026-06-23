# E3b:unseen_sequences_correction 实验计划

## 目标
修正 E3 中测试序列选择的问题：确保用于对比的序列对 **MoBInd 官方 checkpoint** 和 **我们自己的 4-IMU pipeline** 都是 unseen 的。

## 问题回顾
- E3 之前使用的 16 个序列属于 MoBInd 官方 `train` split，因此 MoBInd checkpoint 在训练时见过它们。
- 真正对双方都 unseen 的序列是：**我们 test_sessions 与 MoBInd test/val split 的交集**。

## 选定的 unseen 序列

```
01_002, 03_001, 04_005, 05_002
```

共 4 个序列。

## 子实验

| 编号 | 内容 | 产物 |
|---|---|---|
| B1 | 在该 4-sequence 子集上分别运行 MoBInd 与我们的 4-IMU pipeline（1-window、4-window vote） | `results/mobind_frameacc_unseen.json`, `results/ours_4imu_1window_unseen.json`, `results/ours_4imu_4window_vote_unseen.json` |
| B2 | 汇总对比并生成修正后的结论图表 | `results/results.md`, `results/figures/frameacc_unseen_comparison.png` |

## 验收标准
- 所有模型都在完全相同的 4 个 unseen 序列上输出 FrameAcc。
- 结果表格清晰标明“unseen by both models”。
- 更新 HAROS resume.md，指出 E3 的 train-only subset 存在的偏差。
