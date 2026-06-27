# 长期经验池 (Long-term Insights)

> 记录在本项目中沉淀下来的长线 Insights，避免重复踩坑。

---

## 1. 数据质量优先于算法调优
* **Insight:** 当原始数据中人物动作缺乏区分性时，再复杂的后处理（adaptive decay、Kalman、multi-window voting、confidence calibration）也难以突破上限。
* **证据:** `experiments/scale_test_corrupt/FINAL_RESULTS.md` 显示，将 TotalCapture 任一模态替换为纯噪声后，性能直接跌至 Custom 水平。
* **行动:** 在投入新算法前，先用简单启发式（IMU 方差 + 骨架速度）判断人物是否可区分。

## 2. Greedy + Margin 排序优于 Hungarian
* **Insight:** 在 IMU-Track 分配阶段，按 top-1 margin 排序的贪心分配显著优于匈牙利全局最优。
* **证据:** Custom 上 Greedy 42.8% vs Hungarian ~28%。
* **行动:** 默认使用 greedy assignment，除非有明确证据表明 Hungarian 在特定场景更优。

## 3. Sliding Window Vote 是有效的低成本后处理
* **Insight:** 在全局匹配大致正确的前提下，用覆盖全 session 的滑动窗口多数投票可抑制局部抖动。
* **证据:** Custom 上 42.8% → 51.5%，且对 stride 1–12 不敏感，可 10x 加速推理。
* **行动:** 作为默认后处理；若全局排列错误（如 session 000022），vote 无效，需从模型/数据层面解决。

## 4. Adaptive Decay 的“历史不可信”困境
* **Insight:** 当 baseline 准确率仅 40–50% 时，基于历史分配一致性设计的自适应 decay 会放大错误历史，导致策略失效。
* **证据:** 6 大类 15+ 种 adaptive decay 信号均未超越固定 decay=0.001 基线。
* **行动:** 在 baseline 达到 70%+ 之前，不优先考虑复杂的在线自适应策略。

## 5. IMU 归一化与单位转换收益有限
* **Insight:** 在 TotalCapture 上，per-fold / per-session IMU 归一化、单位转换（g → m/s²）对最终性能影响很小。
* **证据:** `experiments/custom_normalization_4fold/` 与相关 ablation 显示边际或负向收益。
* **行动:** 不要反复在预处理尺度上打转；优先解决数据采集一致性与动作差异性。
