# E6:fair_single_imu_same_window 实时进度日志

## 2026-06-25 22:50
* MoBInd Stage2 训练完成（early stopping，约 34 分钟）。
* MoBInd 评估修复：raw IMU 需取 RightWrist 通道后再输入模型。
* MoBInd single-IMU 24-frame mean FrameAcc = **0.4393**。
* 最终对比：
  * MoBInd single IMU (24-frame): **0.4393**
  * Our pipeline single IMU (1 window): **0.7572**
  * Our pipeline single IMU (4-window vote): **0.7973**
* 生成 `results.md` 与对比图。
* 待完成：更新 resume.md、commit/push。

## 2026-06-25 22:10
* Our pipeline 训练完成（50 epochs，best val top1=0.7161）。
* Our pipeline 评估完成：
  * 1-window mean FrameAcc = **0.7572**
  * 4-window vote mean FrameAcc = **0.7973**
* MoBInd Stage1 训练到 epoch 632 被手动截停（val R@1 ≈ 0.219），best checkpoint 已保存。
* 开始 MoBInd Stage2 训练（stage1_exp 指向 Stage1 实际输出目录，epochs=500，patience=100）。

## 2026-06-25 21:55
* Our pipeline 训练已启动（单 IMU 重复 4 次）。

## 2026-06-25 21:43
* Slice 完成（第二次尝试）：123 序列，13435 个 windows。

## 2026-06-25 21:42
* MoBInd 24 帧单 IMU contrastive cache 生成完成：`cache_action_1.2_0.8`，13427 个窗口。

## 2026-06-25 21:30
* Plan 已审批，选择 Option A：统一使用 24 帧窗口（约 1.2 s @ 20 Hz）。
* 初始化 E6 沙盒目录与 plan.md。
* 开始 A1：准备 MoBInd 单 IMU 24 帧配置与代码补丁。
