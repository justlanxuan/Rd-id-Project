# EgoHumans：Autism-project pipeline vs. MoBInd 官方结果对比

## 1. 对比概览

| 维度 | Autism-project pipeline (E1 A3) | MoBInd official (E2) | 备注 |
|------|-------------------------------|----------------------|------|
| **任务定位** | 跨模态 Re-ID / 关联（IMU ↔ skeleton） | 跨模态检索 + 人体定位 + 时序同步 | 任务集合不同，仅部分指标可对应 |
| **训练数据** | EgoHumans 128 序列（88 train / 20 val / 20 test） | EgoHumans 官方 train split（contrastive cache 4659 样本） | MoBInd 使用官方划分，数据量/划分不同 |
| **输入模态** | MoBInd synthetic IMU (4 sensors) + GT/extracted skeleton | MoBInd synthetic IMU (5 sensors) + pose2d / pose3d | 我们丢弃了 Head sensor，只用了 4 slots |
| **模型架构** | MotionBERT-Lite + SIE_v2 + InfoNCE | ConvFormer + LSTM/Temporal encoder + contrastive stage1/2 | 官方模型专为该任务设计 |
| **训练方式** | 在 EgoHumans 上从头训练，冻结 backbone 50 epochs | 官方预训练权重（stage2 MAE checkpoint） | 官方 checkpoint 已经过更充分训练/预训练 |

## 2. 指标对比

### 2.1 窗口级 / 样本级检索（Retrieval / Top-1）

| 方法 | 指标 | 数值 | 说明 |
|------|------|------|------|
| **MoBInd official** | IMU → Video R@1 | **0.8264** | 全 test gallery（4659 样本）上计算 |
| **MoBInd official** | Video → IMU R@1 | **0.8368** | 全 test gallery 上计算 |
| **Our pipeline (GT skeleton)** | test_top1 | **0.775** | 按 batch 内对角配对计算的 top-1 准确率 |
| **Our pipeline (extracted skeleton)** | test_top1 | **0.730** | 同上，输入为官方 `poses2d/cam03` 转换的 skeleton |

**解读**：
- MoBInd 的 R@1 是在整个 test set 上做最近邻检索，难度高于我们的 per-batch top-1。即便如此，MoBInd 仍高出约 **5–11 个百分点**。
- 这说明在单窗口跨模态检索上，MoBInd 的官方模型表示能力更强。
- 我们使用 extracted skeleton 相比 GT skeleton 下降约 **4.5 pp**，说明 pose 质量对单窗口匹配影响明显。

### 2.2 人体定位 / 身份关联

| 方法 | 指标 | 数值 | 说明 |
|------|------|------|------|
| **MoBInd official** | Person localization | **98.01%** | 多 person 视频中判断哪个人与 IMU 对应 |
| **MoBInd official** | Limb localization | **89.22%** | 在正确 person 上进一步定位肢体 |
| **Our pipeline** | grouped G2/G4/G6/G8 | **100.0%** | 按 person 分组的 chunk-level 匹配准确率 |
| **Our pipeline** | synchronous HOTA | **0.887** | 多 person 时序跟踪指标（AssA=0.880, DetA=0.894） |
| **Our pipeline** | synchronous FrameAcc | **0.956** | 逐帧匹配准确率 |

**解读**：
- MoBInd 的 Person localization 本质上就是 **IMU-to-person identification**：在单个 5 秒窗口内，把 `P` 段 IMU 信号与 `P` 个人的 pose 做 `P × P` 匹配。结果是 **98.01%**，确实非常强。
- 我们的 grouped G2–G8 100% 是**聚合多个窗口后的 chunk-level 匹配**，难度不同，不能直接比较。
- 我们的 **synchronous FrameAcc = 0.956** 与 MoBInd Person localization 在概念上最接近：都是判断“哪段 IMU 属于哪个人”。但两者的评估方式不同：
  - MoBInd：在**单个窗口**内做一次匹配，没有时序上下文，也不处理漏检/轨迹碎片化，输入是官方 pose2d/pose3d。
  - 我们的 FrameAcc：在**整个 sequence 的每一帧**上，用滑动窗口 + Hungarian 分配判断 IMU 与 extracted skeleton track 是否对应，只统计有有效 GT-to-extract 映射的可见帧。
- 因此 **98.01% vs 95.6%** 可以做横向参考，但不是严格同任务指标：MoBInd 的任务更纯粹，我们的任务多了时序连续性和 extracted track 映射带来的噪声。
- 我们没有直接输出 limb-level 定位指标。

### 2.3 时序同步

| 方法 | 指标 | 数值 | 说明 |
|------|------|------|------|
| **MoBInd official** | Sync MAE | **0.0392–0.0421 s** | 预测 IMU 与 video 的时间偏移误差 |
| **MoBInd official** | Acc@0.2 | **0.9925 / 1.0000** | 预测偏移 < 0.2 s 的比例 |
| **Our pipeline** | synchronous HOTA | 0.887 | 时序跟踪指标 |
| **Our pipeline** | synchronous FrameAcc | 0.956 | 逐帧匹配准确率 |

### 2.4 E3 严格对齐的 FrameAcc 对比（推荐参考）

在 `experiments/G_egohumans/E3:mobind_vs_pipeline_frameacc/` 中，我们在**完全相同的 16 个 EgoHumans 序列**上（且均为 MoBInd train split，无 test-set 泄漏）重新计算了 FrameAcc：

| 方法 | 窗口 | 输入 | Mean FrameAcc |
|------|------|------|---------------|
| **MoBInd official** | 5 秒（100 帧） | raw IMU (5 sensors) + COCO pose2d | **0.9654** |
| **Our pipeline** | 1.2 秒（24 帧） | 48-D IMU + H36M skeleton | **0.9562** |

**解读**：
- 这是目前最公平的同任务对比：两者都使用相同的 extract tracks 和 `gt_to_extract_map`。
- MoBInd 领先约 **0.93 pp**，优势很小。
- 说明我们的 pipeline 在 person-level IMU-to-person identification 上已经达到与官方 MoBInd checkpoint 相近的水平。

## 3. 综合判断
- 我们的 pipeline **不显式回归偏移**，而是通过逐帧嵌入匹配完成关联。FrameAcc 95.6% 也不错，但任务定义不同。
- 若目标是“精确估计 IMU 与 video 的时间差”，MoBInd 更优；若目标是“持续保持 IMU 与 skeleton 的身份关联”，我们的跟踪指标也有竞争力。

## 3. 综合判断

| 场景 | 更好的一方 | 原因 |
|------|-----------|------|
| 单窗口跨模态检索 | **MoBInd** | R@1 更高，gallery 规模更大 |
| 单帧人体/肢体定位 | **MoBInd** | Person 98%、Limb 89%，我们的 pipeline 无对应指标 |
| 时间偏移估计 | **MoBInd** | MAE ~40 ms，Acc@0.2 ≈ 100% |
| 多窗口 chunk-level 关联 | **Our pipeline** | grouped G2–G8 100%，说明时序聚合后非常稳定 |
| 可扩展性/可控性 | **Our pipeline** | 是我们自己的代码库，易于接入新数据、新 loss、新评估协议 |

## 4. 我们的 pipeline 为什么总体落后？

1. **训练不足 / 数据量小**：我们仅在 128 序列上训练，且冻结了 MotionBERT backbone；MoBInd checkpoint 是官方调优结果。
2. **任务目标不同**：我们训练的是 Re-ID/关联任务，MoBInd 训练的是更通用的跨模态表示 + 同步 + 定位多任务。
3. **输入差异**：我们只用 4 个 IMU sensors（丢弃 Head），MoBInd 用 5 个，可能损失部分判别信号。
4. **评估方式差异**：我们的 top1 是 per-batch 对角匹配，MoBInd R@1 是全 gallery 检索；即便在这种不对等情况下，MoBInd 仍更高。
5. **架构差异**：MoBInd 的 ConvFormer / Temporal encoder 可能更适合 IMU 时间序列；我们的 MotionBERT backbone 针对 pose 设计，IMU 侧只是 adapter。

## 5. 下一步建议

若想在同一基准上追平或超过 MoBInd：
- **对齐评估**：把我们的模型放到 MoBInd 的 `eval_retrieval.py` / `eval_localization.py` / `eval_sync_egoh.py` 上跑，获得可严格对比的 R@1、localization、sync 指标。
- **改进输入**：保留 Head sensor，或尝试使用全部 5 个 sensors；对 extracted skeleton 做缺失帧插值 / 2D→3D lifting。
- **解冻/微调 backbone**：当前冻结 backbone 限制了表达能力；可尝试分阶段解冻或增加 IMU-specific encoder。
- **引入同步/定位辅助任务**：MoBInd 的多任务训练（retrieval + sync + localization）可能互相促进。
- **数据增强与更大训练集**：若还能获取更多 EgoHumans 序列或其他 ego-centric 数据集，可显著提升泛化。
