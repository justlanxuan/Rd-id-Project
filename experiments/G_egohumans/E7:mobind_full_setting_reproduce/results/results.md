# E7: MoBInd Full-Setting Reproduction — Results

## 1. 实验目标
在 MoBInd 官方默认配置（5 个 IMU 传感器、5 s / 100 帧窗口）下重新训练 Stage1 + Stage2，并在 24 个官方 test 序列上计算 synchronous FrameAcc，验证复现流程是否正确。

## 2. 训练配置
| 项目 | Stage1 | Stage2 |
|---|---|---|
| 配置文件 | `experiments/G_egohumans/E7:mobind_full_setting_reproduce/config/MoBind_stage1.yaml` | `experiments/G_egohumans/E7:mobind_full_setting_reproduce/config/MoBind_stage2.yaml` |
| `window_sec` / `stride_sec` | 5 / 2 | 5 / 2 |
| `multi_sensor` / `num_limbs` | false / — | true / 5 |
| IMU 肢体 | 5: LeftWrist, RightWrist, LeftKnee, RightKnee, Head | 同上 |
| 训练停止 epoch | 5350 (early stopping) | 600 (early stopping) |
| 耗时 | 4.37 h | 0.53 h |
| 最终 val R@1 (motion→imu / imu→motion) | ~0.84 / ~0.81 | ~0.90 / ~0.90 |

### 关键输出路径
* Stage1: `/home/fzliang/MoBind/outputs/EgoHumans/stage1_E7_repro/EgoHumans/06-25-2026:23:13:37`
* Stage2: `/home/fzliang/MoBind/outputs/EgoHumans/stage2_E7_repro/EgoHumans/06-26-2026:11:04:59`
* 评估 JSON: `experiments/G_egohumans/E7:mobind_full_setting_reproduce/results/mobind_retrained_frameacc_aligned_test.json`

## 3. FrameAcc 结果（24 个官方 test 序列）
| sequence | FrameAcc |
|---|---|
| custom_01_001 | 0.9704 |
| custom_01_002 | 0.9789 |
| custom_01_003 | 0.9551 |
| custom_01_004 | 0.9264 |
| custom_03_001 | 0.9804 |
| custom_03_002 | 0.9928 |
| custom_03_003 | 0.9731 |
| custom_03_004 | 0.9817 |
| custom_04_001 | 0.9844 |
| custom_04_002 | 0.9373 |
| custom_04_003 | 0.9691 |
| custom_04_004 | 0.9094 |
| custom_05_001 | 0.9816 |
| custom_05_002 | 0.9660 |
| custom_05_003 | 0.9808 |
| custom_05_004 | 0.9601 |
| custom_06_001 | 0.9362 |
| custom_06_002 | 0.9062 |
| custom_06_003 | 0.9682 |
| custom_06_004 | 0.9701 |
| custom_07_001 | 0.9992 |
| custom_07_002 | 0.9992 |
| custom_07_003 | 1.0000 |
| custom_07_004 | 0.9937 |

**Mean FrameAcc: 0.9675**

## 4. 与官方 checkpoint 对比
* E5 官方 checkpoint（24 test 序列）: **0.9666**
* E7 重新训练（24 test 序列）: **0.9675**
* 差距: **+0.0009**（在随机波动范围内，可认为完全一致）

## 5. 结论
重新训练的 MoBInd 在官方全配置下达到了与官方 checkpoint 相当的 FrameAcc（0.9675 vs 0.9666），说明本仓库对 MoBInd 的训练/评估流程复现是正确的。

因此，E6 中单 IMU + 24 帧窗口下 MoBInd 的 FrameAcc 仅为 0.4393，应归因于**输入设置本身**（单一 IMU、短窗口导致信息不足），而非训练或评估代码 bug。
