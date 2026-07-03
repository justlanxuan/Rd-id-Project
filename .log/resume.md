# 🔄 HAROS Task Resume Node

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G4:mobind_single_imu_adaptation — **新目标：探索 MoBInd 单 IMU 最优适配方案**。
* **上一阶段收尾:** G_egohumans — E6 cache bug 已修复，结果已反转；G3/E2 换 seed 复现完成，显示 custom 上单 IMU MoBInd seed 方差极大。
* **当前具体执行任务:**
  - ✅ G4 目录与 HAROS 文件已初始化（formulation / survey / ideas / plan / E1）。
  - 🔄 E1: 整理 w24/w100 6-seed 基线结果，建立统一评估框架。
* **当前具体执行任务:**
  - ✅ E9（MoBInd 5 IMU / 24 帧窗口）已完成，FrameAcc = **0.9641**。
  - ✅ E6-correct（MoBInd 单 IMU / 24 帧，cache bug 修复后）已完成，FrameAcc = **0.9548**。
* **当前子任务状态:**
  - ✅ E6-correct：MoBInd 单 IMU / 24 帧 = **0.9548**（cache bug 修复后，valid）。
  - ❌ E6-original：0.4393（受 cache bug 影响，已废弃）。
  - ❌ E6-aligned：0.4556（仍受 cache bug 影响，已废弃）。
  - ✅ E7：MoBInd 5 IMU / 100 帧 = 0.9675
  - ✅ E8：MoBInd 单 IMU / 100 帧 = 0.9616
  - ✅ E9：MoBInd 5 IMU / 24 帧 = 0.9641
  - ✅ G3/E1：Autism pipeline 全 IMU / 自提取骨架 / 24 帧 = 0.942
  - ✅ G3/E1：Autism pipeline 单 IMU / 自提取骨架 / 24 帧 = 0.677

## 2. 最终结果与结论
### MoBInd 控制变量（窗口 vs IMU 数量，修正后）
| 实验 | IMU | 窗口 | FrameAcc | 备注 |
|---|---|---|---|---|
| E6-correct | 1 IMU | 24 帧 | **0.9548** | ✅ cache bug 修复后 |
| E9 | 5 IMU | 24 帧 | **0.9641** | — |
| E8 | 1 IMU | 100 帧 | **0.9616** | — |
| E7 | 5 IMU | 100 帧 | **0.9675** | — |

- **原始 E6 低性能完全由 cache bug 导致**，并非“单 IMU + 24 帧不可行”。
- 修正后，MoBInd 在四种配置下均接近 0.96，说明其对窗口长度和 IMU 数量均高度鲁棒（在同关节 IMU↔pose 匹配任务上）。
- 单 IMU / 24 帧（0.9548）与 5 IMU / 24 帧（0.9641）仅差 **0.9 pp**；与单 IMU / 100 帧（0.9616）仅差 **0.7 pp**。
- **重要限制**：E6/E8 中 MoBInd 的 `num_limbs=1` 同时把视频侧 motion 限制为同一个肢体，做的是“同关节 IMU ↔ pose”匹配，而非真正的“单 IMU + 全视频”匹配。

### MoBInd vs Autism-project pipeline（24 帧窗口）
| 方法 | IMU | FrameAcc |
|---|---|---|
| MoBInd（E9） | 5 IMU | **0.9641** |
| MoBInd（E6-correct） | 1 IMU | **0.9548** |
| Autism pipeline | 全 IMU（4） | **0.942** |
| Autism pipeline | 单 IMU（R_LowArm） | **0.677** |

- MoBInd 单 IMU / 24 帧即可超过 Autism pipeline 全 IMU / 24 帧。
- MoBInd 对 IMU 数量的敏感度远低于 Autism pipeline：
  - MoBInd 5 IMU → 1 IMU：-0.9 pp
  - Autism 4 IMU → 1 IMU：-26.5 pp

## 3. 根因记录
- 文件：`experiments/G_egohumans/E6:fair_single_imu_same_window/diagnosis.md`
- 核心 bug：`preprocess/EgoHumans/cache.py` 用 `enumerate(limb_list)` 索引 `imu_data`。当 `limb_list = ["RightWrist"]` 时，实际保存的是 `imu_data[:, 0]`（`LeftWrist`），但文件名为 `RightWrist`。
- 修复：在 `cache.py` 中加入固定肢体顺序映射，按名称取索引。
- 重新生成 cache 后，前 10 窗口与 raw 数据 max diff = 0.0。

## 4. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* 历史错误结果（E6-original / E6-aligned）已在 `control_variable_summary.md/json` 中标记为 `invalidated`。

## 5. 下一步行动 (Next Actions)
### G4 当前
* [ ] 完成 E1 baseline 结果整理（6 seeds 汇总、results.md、可视化）。
* [ ] 创建 E1 一键评估脚本 `scripts/eval_all_seeds.sh`。
* [ ] 人类确认 G4 路线图中优先验证的假设（建议优先 I1：单 IMU + 全骨架）。

### 历史待办
* [x] 检查 `MoBind/configs/config.py` 已恢复为 5-limb。
* [x] 清理旧 cache 与错误 Stage1/Stage2 checkpoint。
* [ ] 更新 Autism-project 主 README / 论文表格中的 E6 数值与结论。
