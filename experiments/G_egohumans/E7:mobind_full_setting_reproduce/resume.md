# E7:mobind_full_setting_reproduce — Resume

## 1. 实验目标
用 MoBInd 官方原生配置（5 IMU、5 s / 100 帧窗口）重新训练并评估，验证 MoBInd 复现流程的正确性。

## 2. 最终结论
* **Mean FrameAcc (24 test sequences): 0.9675**
* 与 E5 官方 checkpoint 的 0.9666 基本一致（差距 < 0.001）。
* **复现成功**：MoBInd 训练/评估流程无 bug。
* E6 中低性能（0.4393）归因于单 IMU + 24 帧输入设置，而非代码问题。

## 3. 关键结果文件
* `results/results.md`
* `results/mobind_retrained_frameacc_aligned_test.json`

## 4. 关键训练输出
* Stage1: `/home/fzliang/MoBind/outputs/EgoHumans/stage1_E7_repro/EgoHumans/06-25-2026:23:13:37`
* Stage2: `/home/fzliang/MoBind/outputs/EgoHumans/stage2_E7_repro/EgoHumans/06-26-2026:11:04:59`

## 5. 下一步建议
* 将 E7 结果连同 `.log/resume.md` 一起提交到 `egohumans` 分支。
* 在论文/报告中引用 E7 作为“MoBInd 复现正确性”的支撑证据。
