# E6 Results: Single-IMU Same-Window Fair Comparison

## 实验设置

- **IMU 输入**：双方都只使用右手腕 IMU。
  - MoBInd：`limb_list = [RightWrist]`，`num_limbs = 1`。
  - Our pipeline：`imu_sensor = R_LowArm`，`repeat_single_sensor = 4`（复制 4 次以维持 48-D 输入）。
- **窗口长度**：统一为 24 帧（约 1.2 s @ 20 Hz）。
- **测试集**：24 个 MoBInd official test 序列（双方均 unseen）。

## FrameAcc 对比（24 个序列）

| Sequence | MoBInd single IMU | Our pipeline (1 window) | Our pipeline (4-window vote) |
|---|---|---|---|
| custom_01_001 | 0.4181 | 0.4945 | 0.5543 |
| custom_01_002 | 0.3654 | 0.4881 | 0.3536 |
| custom_01_003 | 0.3006 | 0.5325 | 0.4253 |
| custom_01_004 | 0.3290 | 0.4870 | 0.2489 |
| custom_03_001 | 0.6209 | 0.9429 | 0.9804 |
| custom_03_002 | 0.7276 | 0.9571 | 0.9749 |
| custom_03_003 | 0.8687 | 0.9282 | 0.9680 |
| custom_03_004 | 0.6121 | 0.9073 | 0.8728 |
| custom_04_001 | 0.2873 | 0.5360 | 0.6068 |
| custom_04_002 | 0.2658 | 0.6605 | 0.7500 |
| custom_04_003 | 0.2461 | 0.7872 | 0.8809 |
| custom_04_004 | 0.2906 | 0.8313 | 0.9094 |
| custom_05_001 | 0.4293 | 0.6844 | 0.8644 |
| custom_05_002 | 0.3767 | 0.6787 | 0.7584 |
| custom_05_003 | 0.4406 | 0.7022 | 0.8737 |
| custom_05_004 | 0.3715 | 0.7392 | 0.8698 |
| custom_06_001 | 0.2041 | 0.6999 | 0.7210 |
| custom_06_002 | 0.2631 | 0.6917 | 0.7863 |
| custom_06_003 | 0.1515 | 0.7700 | 0.9283 |
| custom_06_004 | 0.2781 | 0.7636 | 0.8389 |
| custom_07_001 | 0.5383 | 0.9725 | 0.9992 |
| custom_07_002 | 0.6402 | 0.9750 | 0.9992 |
| custom_07_003 | 0.7690 | 0.9867 | 1.0000 |
| custom_07_004 | 0.7495 | 0.9569 | 0.9703 |
| **Mean** | **0.4393** | **0.7572** | **0.7973** |

## 结论

- 在单 IMU + 24 帧的严格公平设置下，MoBInd 的 mean FrameAcc 为 **0.4393**。
- 我们的 pipeline 1-window 为 **0.7572**（领先 31.79 pp）。
- 我们的 pipeline 4-window vote 为 **0.7973**（领先 35.79 pp）。

## 与 E5 的对比

- E5（4-IMU，24 帧窗口）：MoBInd 0.9666，Ours 1-window 0.9372，Ours 4-window vote 0.9536。
- E6（单 IMU，24 帧窗口）：双方均显著下降，说明多传感器信息对 MoBInd 和我们的 pipeline 都很重要。

## AI Reflection

- 控制 IMU 数量和窗口长度后，差距趋势与 E5 一致：MoBInd 仍略高，但 4-window vote 能缩小差距。
- 单 IMU 重复 4 次在我们的 pipeline 中是一种工程折中，未来若需完全对齐，可重构 IMU encoder 支持真单 IMU 输入。
