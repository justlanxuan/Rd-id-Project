# 🔄 HAROS Task Resume Node

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G4:mobind_single_imu_adaptation — **E11: Dual IMU Embedding（local + global）分别训练再融合 — 已完成，A8 统计显著性检验已完成**。
* **上一阶段收尾:** G_egohumans — E6 cache bug 已修复，结果已反转；G3/E2 换 seed 复现完成，显示 custom 上单 IMU MoBInd seed 方差极大。
* **当前具体执行任务:**
  - ✅ G4 目录与 HAROS 文件已初始化（formulation / survey / ideas / plan / E1 / E11 / A6 / A8）。
  - ✅ E1: 6-seed 基线已整理完成（w24 0.673±0.182，w100 0.659±0.130）。
  - ✅ E11-A1/A2: w24/w100 Model-G 训练与评估完成。
  - ✅ E11-A3/A4: 双模型融合与聚合完成，`results/results.md` 已生成。
  - ✅ E11-A6/A7/A8: per-frame 可视化与统计检验完成，`results/per_frame_analysis.md` 与 `vis/per_frame_analysis/` 已生成。
  - ⏳ 等待人类决策：是否提交 Git、是否更新 SOTA、是否继续 E12/E13。

## 2. 最终结果与结论
### E11 w24 结果（6 seeds, sim_norm=none）
| 方法 | Mean FrameAcc | Std |
|---|---|---|
| Model-L (local) | 0.673 | 0.183 |
| Model-G (global) | 0.664 | 0.090 |
| Fusion α=0.5 | 0.709 | 0.124 |
| **Fusion best α** | **0.752** | **0.095** |

### E11 w100 结果（6 seeds, sim_norm=none）
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

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* 注意：当前 `.log/resume.md`、E11 `progress.md`、E11 `results/*.md`、E11 `vis/per_frame_analysis/`、E11 w100 配置文件均为未提交修改（`experiments/` 与 `vis/` 在 `.gitignore` 中，仅 `.log/resume.md` 会被 Git 追踪）。

## 4. 下一步行动 (Next Actions)
* [ ] 人类决策：是否将当前 `.log/resume.md` 提交 Git Commit。
* [ ] 人类决策：是否将 w24 Fusion best α（0.752 ± 0.095）更新为项目 SOTA。
* [ ] 人类决策：是否继续推进 E12（课程式 local→global 训练）或 E13（自适应 local/global 门控）。
