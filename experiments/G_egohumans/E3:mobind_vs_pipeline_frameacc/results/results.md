# E3 Results: MoBInd vs. Pipeline FrameAcc on EgoHumans

## 目标
在完全相同的 16 个 EgoHumans 序列上，用 **MoBInd 官方 checkpoint** 和 **我们自己的 trained pipeline** 分别计算 IMU-to-person identification 的 **FrameAcc**，并保证 MoBInd 没有在测试序列上训练过。

## 测试序列
仅使用 MoBInd 官方 train split 中的 16 个序列，排除与 MoBInd test/val 重叠的 4 个序列：

```
01_011, 02_001, 03_009, 04_011, 05_007,
06_024, 06_040, 06_041, 06_054, 06_019,
06_036, 06_006, 06_025, 06_060, 07_011, 07_007
```

被排除的序列：
- 在 MoBInd official test 中：`01_002`, `03_001`, `05_002`
- 在 MoBInd official val 中：`04_005`

## 方法

### MoBInd (A1)
- 脚本：`scripts/A1_eval_mobind_frameacc.py`
- 模型：`/home/fzliang/MoBind/checkpoints/EgoHumans/stage2/best.pt`（stage2 MAE，加载 stage1）
- 输入：MoBInd raw IMU `(T, 5, 7)` + E1 A3 提取的 `skeleton.json` 中的 COCO-17 pose2d
- 窗口：5 秒（100 帧），stride 16 帧（因 ConvFormer 固定 5 秒窗口）
- 分配：每窗口用匈牙利算法将 `N_imu` 段 IMU 分配给活跃 extract tracks

### Our Pipeline (A2)
- 脚本：`scripts/A2_eval_ours_frameacc_subset.py`
- 模型：`data/interim/egohumans_full_extract/train/egohumans_full_extract/best.pt`
- 输入：pipeline NPZ（48-D IMU + root-relative scale-normalized H36M skeleton）
- 窗口：24 帧，stride 16 帧（与训练配置一致）
- 分配：`src/engine/eval_synchronous.py` 的逐帧 Hungarian 匹配

### 对比 (A3)
- 脚本：`scripts/A3_compare_results.py`
- 输出：`results/figures/frameacc_comparison.png`

## 结果

| Sequence | MoBInd FrameAcc | Our Pipeline FrameAcc | Δ (MoBInd - Ours) |
|----------|----------------|----------------------|-------------------|
| custom_01_011 | 0.9329 | 0.9184 | +0.0145 |
| custom_02_001 | 0.9731 | 0.9360 | +0.0371 |
| custom_03_009 | 0.9922 | 0.9922 | 0.0000 |
| custom_04_011 | 1.0000 | 1.0000 | 0.0000 |
| custom_05_007 | 0.9784 | 0.9072 | +0.0712 |
| custom_06_006 | 0.9260 | 0.9226 | +0.0033 |
| custom_06_019 | 0.9846 | 0.9846 | 0.0000 |
| custom_06_024 | 0.9326 | 0.9326 | 0.0000 |
| custom_06_025 | 0.9733 | 0.9633 | +0.0100 |
| custom_06_036 | 0.9443 | 0.9305 | +0.0137 |
| custom_06_040 | 0.9969 | 0.9969 | 0.0000 |
| custom_06_041 | 0.9677 | 0.9677 | 0.0000 |
| custom_06_054 | 0.9637 | 0.9637 | 0.0000 |
| custom_06_060 | 0.8840 | 0.8854 | -0.0014 |
| custom_07_007 | 0.9987 | 0.9987 | 0.0000 |
| custom_07_011 | 0.9987 | 0.9987 | 0.0000 |
| **Mean** | **0.9654** | **0.9562** | **+0.0093** |

![FrameAcc comparison](figures/frameacc_comparison.png)

## 结论
- **MoBInd 略胜一筹**：在相同的 16 个序列上，MoBInd 的 mean FrameAcc 为 **96.54%**，我们的 pipeline 为 **95.62%**，差距约 **0.93 个百分点**。
- **序列级差异**：MoBInd 在 `custom_05_007`（+7.1%）和 `custom_02_001`（+3.7%）上优势明显；其他序列基本持平，仅在 `custom_06_060` 上我们略高（-0.14%）。
- **总体判断**：两个模型在 IMU-to-person identification 任务上表现非常接近，MoBInd 的官方 checkpoint 稍强，但我们的 pipeline 在只训练了 128 序列、冻结 backbone 的条件下已经具有很强的竞争力。

## 重要 Caveat
- **窗口长度不同**：MoBInd 因模型结构必须使用 5 秒窗口（100 帧），我们的模型使用训练时的 1.2 秒窗口（24 帧）。这是在不重训任一模型的情况下能做到的最公平对比。
- **输入预处理不同**：MoBInd 使用 raw IMU + 像素归一化 COCO pose2d；我们的 pipeline 使用转换后的 48-D IMU + root-relative scale-normalized H36M skeleton。虽然 FrameAcc 定义相同，但特征域不同。
- **本实验仅对比 person-level identification（FrameAcc）**，不包含 MoBInd 的 limb localization。
