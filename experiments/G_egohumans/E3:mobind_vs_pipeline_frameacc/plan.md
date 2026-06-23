# E3: Strict MoBInd vs. Pipeline FrameAcc Comparison on EgoHumans

## 目标
在完全相同的 EgoHumans 测试序列上，用 **官方 MoBInd checkpoint** 和 **我们自己的 trained pipeline** 分别计算 IMU-to-person identification 的 **FrameAcc**，并严格保证 MoBInd 没有在测试序列上训练过。

## 背景
- MoBInd 最接近的任务是 `eval_localization.py --task person`（Person localization），但它是按 5 秒 clip 做 `P × P` 匹配。
- 我们自己的任务是 `src/engine/eval_synchronous.py` 里的逐帧 Hungarian 匹配，输出 FrameAcc。
- 两个指标概念相近但实现不同；本实验对齐它们 as much as possible。
- 额外探索：由于我们的窗口（24 帧，~1.2s）远小于 MoBInd（100 帧，5s），测试了三种聚合方式来匹配 MoBInd 的决策粒度：
  1. 单窗口 baseline（A2）
  2. 4 窗口 embedding 平均（A4）
  3. 4 窗口 assignment 投票（A5）

## 子实验

### A1: MoBInd FrameAcc
- 脚本：`scripts/A1_eval_mobind_frameacc.py`
- 输入：E1 A3 生成的 NPZ + skeleton.json + MoBInd raw extracted_data `.npy`
- 模型：`/home/fzliang/MoBind/checkpoints/EgoHumans/stage2/best.pt`
- 窗口：5 秒（100 帧），stride 16 帧
- 输出：`results/mobind_frameacc.json`

### A2: Our Pipeline FrameAcc — single window
- 脚本：`scripts/A2_eval_ours_frameacc_subset.py`
- 输入：过滤后的 `windows_test.csv`
- 模型：`data/interim/egohumans_full_extract/train/egohumans_full_extract/best.pt`
- 窗口：24 帧，stride 16 帧
- 输出：`results/ours_frameacc.json`

### A4: Our Pipeline FrameAcc — 4 windows embedding mean
- 脚本：`scripts/A4_eval_ours_frameacc_4window.py`
- 每连续 4 个窗口的 **embedding 平均**后共同做一次 Hungarian 决策。
- 输出：`results/ours_frameacc_4window.json`

### A5: Our Pipeline FrameAcc — 4 windows assignment vote
- 脚本：`scripts/A5_eval_ours_frameacc_4window_vote.py`
- 每连续 4 个窗口各自先做 Hungarian，然后对 **assignment 做多数投票**。
- 输出：`results/ours_frameacc_4window_vote.json`

### A3: 对比与可视化
- 脚本：`scripts/A3_compare_results.py`
- 输出：`results/results.md` + `results/figures/frameacc_comparison.png`

## 测试序列
仅使用 MoBInd 官方 train split 中的 16 个序列，排除与 MoBInd test/val 重叠的 4 个序列：

```
01_011, 02_001, 03_009, 04_011, 05_007,
06_024, 06_040, 06_041, 06_054, 06_019,
06_036, 06_006, 06_025, 06_060, 07_011, 07_007
```

## 验收标准
- A1/A2/A4/A5 均成功运行并在 16 个序列上输出 mean FrameAcc。
- A3 生成包含 1-window、4-window-mean、4-window-vote 的对比表格与图表。
- HAROS 文档同步更新。
