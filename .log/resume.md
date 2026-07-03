# 🔄 HAROS Task Resume Node

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G5:cross_dataset_transfer — **E1: EgoHumans 预训练 dual-embedding 迁移到 custom — 刚初始化，等待开始**。
* **上一阶段收尾:** G4:mobind_single_imu_adaptation — E11 dual-embedding 完成，A8 统计检验完成，SOTA 已更新为 w24 Fusion best α 0.752 ± 0.095（Commit: `dd61608`）。
* **当前具体执行任务:**
  - ✅ G5 goal 已初始化（formulation / survey / ideas / plan / progress）。
  - ✅ E1 实验目录已创建（formulation / plan / progress / scripts / configs）。
  - ✅ A1: EgoHumans cache 已构建（`cache_action_1.2_0.5`，21,179 windows）。
  - ✅ A3: Source Model-G seeds 0/1 已完成；seeds 42/123 训练中。
  - ✅ A2: Source Model-L seed0 已完成；seed42 训练中。
  - ✅ A5: Seed0 zero-shot 完成：Mean FrameAcc = **0.2940**。
  - ✅ A6: Seed0 Local fine-tune 完成：Mean FrameAcc = **0.7259**（接近 from-scratch 0.752）。
  - 🔄 A6: Seed0 Global fine-tune 运行中（GPU 7）。

## 2. 最终结果与结论
### G4/E11 w24 结果（6 seeds, sim_norm=none）
| 方法 | Mean FrameAcc | Std |
|---|---|---|
| Model-L (local) | 0.673 | 0.183 |
| Model-G (global) | 0.664 | 0.090 |
| Fusion α=0.5 | 0.709 | 0.124 |
| **Fusion best α** | **0.752** | **0.095** |

### G4/E11 w100 结果（6 seeds, sim_norm=none）
| 方法 | Mean FrameAcc | Std |
|---|---|---|
| Model-L (local) | 0.659 | 0.130 |
| Model-G (global) | 0.641 | 0.196 |
| Fusion α=0.5 | 0.675 | 0.135 |
| **Fusion best α** | **0.723** | **0.124** |

### A8 统计显著性检验结论（90,330 有效帧）
- **Local 与 Global 错误显著正相关，而非独立**：
  - w24: χ² = 13,993.74, p ≈ 0；w100: χ² = 18,546.71, p ≈ 0。
- **实际互补帧比例低于独立随机期望**：
  - w24: L-only 14.3% vs 期望 22.8%（比值 0.63）；G-only 12.8% vs 期望 21.3%（比值 0.60）。
  - w100: L-only 13.4% vs 期望 23.4%（比值 0.57）；G-only 11.8% vs 期望 21.8%（比值 0.54）。
- **但 L-only / G-only 帧具有显著时序结构**：
  - w24: L-only 平均连续段 39.2 帧，G-only 30.1 帧（随机 baseline ~1.16 帧）。
  - w100: L-only 23.4 帧，G-only 35.8 帧（随机 baseline ~1.15 帧）。
- **Fusion 仍然可靠地 rescue 了大部分单一模型错误帧**：
  - w24: 27.1% 的帧通过融合从单一模型错误变为正确；w100: 25.2%。
  - 极少出现单一模型对但 Fusion 错的情况（~1.4–1.7%）。

**综合判断**：Fusion 的收益是真实的，但 Local 与 Global 的互补性被高估——它们的错误模式高度相关，导致“只有一个模型对”的帧比理想独立专家少约 40–45%。Fusion 的上限因此受限，未来需要关注如何处理两者共同失败的片段。

### G5 新目标
- **核心问题：** 如何通过在其他数据集（EgoHumans）上预训练，实现在 custom 上的 zero-shot 提升或 fine-tune 提升？
- **第一个实验 E1：** 将 G4/E11 的 local + global dual-embedding 架构先在 EgoHumans 上训练，再迁移到 custom。

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* 注意：`.log/resume.md` 与 `experiments/SOTA_reproduce.md` 已提交至 Git；G5/E1 文档在 `experiments/` 目录下，按 `.gitignore` 不进入 Git，但本次通过 `git add -f` 可显式提交 goal 级文档。

## 4. 下一步行动 (Next Actions)
* [x] G5 goal 文档已提交 Git（Commit: `d10055c`）。
* [x] E1-A1 EgoHumans cache 已构建完成。
* [🔄] E1-A2/A3 source 训练 seed0 验证中。
* [ ] 待 seed0 成功后，扩展至 6 seeds 并行训练。
