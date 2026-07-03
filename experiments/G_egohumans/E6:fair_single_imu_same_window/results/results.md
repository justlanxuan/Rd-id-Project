# E6 Results: Single-IMU Same-Window Fair Comparison

## 实验设置

- **IMU 输入**：双方都只使用右手腕 IMU。
  - MoBInd：`limb_list = [RightWrist]`，`num_limbs = 1`。
  - Our pipeline：`imu_sensor = R_LowArm`，`repeat_single_sensor = 4`（复制 4 次以维持 48-D 输入）。
- **窗口长度**：统一为 24 帧（约 1.2 s @ 20 Hz）。
- **测试集**：24 个 MoBInd official test 序列（双方均 unseen）。

## 重要更正：cache bug 修复

原始 E6 报告 MoBInd single-IMU FrameAcc = **0.4393**，随后对齐 Stage2 预算后得到 **0.4556**。经 HAROS 根因排查，发现 `preprocess/EgoHumans/cache.py` 在生成单 IMU cache 时存在索引映射 bug：当 `limb_list` 临时改为 `["RightWrist"]` 时，脚本仍按 `enumerate(limb_list)` 取 `imu_data[:, 0]`，实际保存的是 `LeftWrist`，但文件名标为 `RightWrist`。这导致训练与测试的 IMU 通道不一致。

已修复 `cache.py`（按固定肢体顺序映射名称到索引），重新生成 cache 并完整重训 Stage1 + Stage2。修正后的结果如下：

## FrameAcc 对比（24 个序列，MoBInd 已修正）

| Sequence | MoBInd single IMU (corrected) | Our pipeline (1 window) | Our pipeline (4-window vote) |
|---|---|---|---|
| custom_01_001 | 0.9417 | 0.4945 | 0.5543 |
| custom_01_002 | 0.9789 | 0.4881 | 0.3536 |
| custom_01_003 | 0.8799 | 0.5325 | 0.4253 |
| custom_01_004 | 0.9264 | 0.4870 | 0.2489 |
| custom_03_001 | 0.9804 | 0.9429 | 0.9804 |
| custom_03_002 | 0.9928 | 0.9571 | 0.9749 |
| custom_03_003 | 0.9731 | 0.9282 | 0.9680 |
| custom_03_004 | 0.9817 | 0.9073 | 0.8728 |
| custom_04_001 | 0.9631 | 0.5360 | 0.6068 |
| custom_04_002 | 0.9004 | 0.6605 | 0.7500 |
| custom_04_003 | 0.9224 | 0.7872 | 0.8809 |
| custom_04_004 | 0.9094 | 0.8313 | 0.9094 |
| custom_05_001 | 0.9570 | 0.6844 | 0.8644 |
| custom_05_002 | 0.9660 | 0.6787 | 0.7584 |
| custom_05_003 | 0.9297 | 0.7022 | 0.8737 |
| custom_05_004 | 0.9601 | 0.7392 | 0.8698 |
| custom_06_001 | 0.9269 | 0.6999 | 0.7210 |
| custom_06_002 | 0.9045 | 0.6917 | 0.7863 |
| custom_06_003 | 0.9682 | 0.7700 | 0.9283 |
| custom_06_004 | 0.9701 | 0.7636 | 0.8389 |
| custom_07_001 | 0.9992 | 0.9725 | 0.9992 |
| custom_07_002 | 0.9992 | 0.9750 | 0.9992 |
| custom_07_003 | 1.0000 | 0.9867 | 1.0000 |
| custom_07_004 | 0.9837 | 0.9569 | 0.9703 |
| **Mean** | **0.9548** | **0.7572** | **0.7973** |

## 结论

- 修正 cache bug 后，MoBInd 单 IMU + 24 帧的 mean FrameAcc 为 **0.9548**，与 E8（单 IMU / 100 帧，0.9616）和 E9（5 IMU / 24 帧，0.9641）处于同一水平。
- 因此，**原始 E6 的“单 IMU + 24 帧不可行”结论被推翻**，低性能完全由 cache 污染导致。
- 在公平设置下，MoBInd 单 IMU 显著优于 Our pipeline 单 IMU（0.9548 vs 0.7572，领先 19.76 pp）。
- Our pipeline 4-window vote（0.7973）仍不及 MoBInd 单 IMU。

## 重要限制：这不是“单 IMU + 全视频”的测试

MoBInd 的 `ContrastiveMAE` 在 Stage2 中要求 IMU 与 motion 输入具有相同的 `num_limbs`。当我们把 `num_limbs` 设为 1 时，**视频侧（motion）也只会使用 RightWrist 这一个肢体的 pose2d 运动**，而不是完整骨架。因此：
- E6/E8 的 MoBInd 实际上是 **RightWrist IMU ↔ RightWrist pose** 的匹配。
- Our pipeline 的视频侧使用完整骨架（所有关节），因此二者在视频输入上并不完全对等。
- 在 EgoHumans 这种 synthetic、低噪声数据上，匹配同一个关节的 IMU 与 pose 轨迹非常容易，这是单 IMU 也能达到 0.95 的主要原因之一。
- 若要做真正严格的“单 IMU + 全视频”对比，需要修改 MoBInd 以支持 motion 用 5 limb、IMU 用 1 limb（例如对其他 IMU slot 补零或加 mask），这是当前架构未支持的。

## 历史错误值（仅供追溯）

| 版本 | MoBInd single IMU FrameAcc | 说明 |
|---|---|---|
| E6-original | 0.4393 | Stage2 500 epochs，受 cache bug 影响 |
| E6-aligned | 0.4556 | Stage2 10000 epochs，仍受 cache bug 影响 |
| E6-correct | **0.9548** | cache bug 修复后，但 motion 侧也是单 limb |

## 与 E5 的对比

- E5（4-IMU，24 帧窗口，全骨架视频）：MoBInd 0.9666，Ours 1-window 0.9372，Ours 4-window vote 0.9536。
- E6-correct（单 IMU，24 帧窗口，MoBInd 视频侧也只用 RightWrist）：MoBInd 0.9548，Ours 1-window 0.7572，Ours 4-window vote 0.7973。
- 对 MoBInd 而言，从 4 IMU 降到 1 IMU（且 motion 也降到单 limb）仅下降约 1.2 pp。
- 对 Our pipeline 而言，从 4 IMU 降到 1 IMU 下降约 18 pp，说明其更依赖多传感器输入。

## AI Reflection

- 控制变量实验必须首先保证训练/测试数据完全一致；cache 生成脚本中的 `enumerate(limb_list)` 是一个容易被忽视的 bug。
- 修正 cache bug 后，MoBInd 在“单 IMU + 单 limb motion”设置下仍能保持高性能，说明其 contrastive + MAE 预训练对稀疏同模态信号的拟合能力极强。
- 但需要注意：**这个高数字不能简单外推到“真实场景下用单 IMU 做全视频人物匹配”**；EgoHumans synthetic 数据 + 同关节 motion 输入使任务变得简单。
- 后续若继续优化 Our pipeline，可考虑引入自监督预训练、多窗口聚合，或设计支持单 IMU + 全视频的架构。
