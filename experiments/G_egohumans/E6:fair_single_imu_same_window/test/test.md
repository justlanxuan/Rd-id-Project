# E6 Test Plan

## 测试目标
验证在统一单 IMU（右手腕）与统一 24 帧窗口条件下，MoBInd 与 our pipeline 均能成功训练并在 24 个 test 序列上得到有效的 synchronous FrameAcc。

## 测试对象
- MoBInd Stage1 + Stage2（单 IMU、24 帧窗口）
- Our pipeline（`imu_sensor=R_LowArm`、`repeat_single_sensor=4`、24 帧窗口）

## 性能指标
- mean FrameAcc on 24 MoBInd official test sequences
- 1-window 与 4-window vote（仅 our pipeline）

## 通过阈值
- 双方均不出现 NaN / 崩溃。
- Our pipeline 单 IMU FrameAcc 应低于 4-IMU E5 结果（0.9372），但应在合理范围（>0.70）。
- MoBInd 单 IMU FrameAcc 应低于 5-IMU E5 结果（0.9666），但应在合理范围（>0.80）。

## 复现实验步骤
1. 应用 MoBInd 补丁（单 IMU limb_list + 24 帧窗口 + build_model 传参）。
2. 生成 MoBInd 24 帧 contrastive cache。
3. 训练 MoBInd Stage1 → Stage2。
4. 训练 our pipeline 单 IMU 模型。
5. 双方同步评估并生成对比表。

## 失败判定标准
- 任一阶段训练崩溃或 loss 不下降。
- 同步评估出现维度/索引错误。
- 最终 FrameAcc 为 NaN。
