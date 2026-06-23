# E5:mobind_aligned_splits 实验计划

## 目标
使用与 MoBInd 官方完全相同的 train/val/test 划分重新训练我们的 4-IMU pipeline，并在 MoBInd 官方 test set（24 个序列）上与 MoBInd 官方 checkpoint 做公平对比。

## 划分来源
MoBInd 官方 `configs/config.py` 中的 `EgoHumans.split.action`：

| Split | 序列数 | 序列来源 |
|---|---|---|
| train | 98 | MoBInd action split train |
| val | 6 | MoBInd action split val |
| test | 24 | MoBInd action split test |

## 子实验

| 编号 | 内容 | 产物 |
|---|---|---|
| A1 | 准备对齐配置 `config/egohumans_mobind_aligned.yaml` | 配置文件 |
| A2 | 运行 extract + slice 阶段 | `data/interim/egohumans_mobind_aligned/slice/windows_*.csv` |
| A3 | 训练 4-IMU 模型 | `data/interim/egohumans_mobind_aligned/train/egohumans_mobind_aligned/best.pt` |
| A4 | 在 MoBInd test set 上评估我们的模型 | `results/ours_mobind_aligned_1window.json`, `results/ours_mobind_aligned_4window_vote.json` |
| A5 | 在相同 test set 上评估 MoBInd | `results/mobind_mobind_aligned_test.json` |
| A6 | 汇总对比并生成结果文档 | `results/results.md`, `results/figures/frameacc_mobind_aligned.png` |

## 关键控制变量
- 除数据划分外，所有参数与 `egohumans_full_extract.yaml` 保持一致。
- 使用相同的 4-IMU 输入、24 帧窗口、50 epochs、frozen backbone。
- MoBInd 仍使用原生 5 秒窗口；我们报告 1-window 与 4-window vote。
