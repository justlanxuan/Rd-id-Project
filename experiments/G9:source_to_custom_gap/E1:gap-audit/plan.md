# E1：骨架源与输入 gap 审计

## 目标

在任何新的训练或适配实验前，确认所有骨架源的真实内容、表示空间、质量、provenance 和 source/Custom 可用性。

## 输入

- G6 TotalCapture、EgoHumans、Custom manifest；
- S06 algorithm outputs；
- AlphaPose、YOLO-Pose high、WHAM 原始输出；
- EgoHumans pose2d caches；
- 待验证后端仅记录可用性，不直接进入正式结果。

## 步骤

1. 生成 source inventory 和 stable content hash；
2. 校验 H36M-17 mapping、shape、finite、时间单调性；
3. 计算 joint confidence、missing、bone stability、tracklet 统计；
4. 对疑似同源输出计算 pairwise correlation 和差异；
5. 生成 source availability matrix；
6. 选择可进入 E2 的正式源并记录排除理由；
7. 更新 E1 `progress.md` 和 `results/results.md`。

## 完成标准

- 无静默空结果或无 provenance 的正式 artifact；
- 2D、3D、SMPL 分轨；
- 每个进入正式矩阵的源可从 manifest 追溯到原始文件和配置；
- 测试文件中定义的所有失败样例均能 fail-loud。
