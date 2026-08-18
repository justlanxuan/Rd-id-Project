# E1 Test Contract：骨架源审计

## 测试目标

验证所有进入 G9 的骨架 artifact 具有可追溯 provenance、正确 joint mapping、有限值、正确时间结构，并且不会将重复或空结果伪装成独立算法。

## 测试对象

- G6 canonical NPZ/CSV；
- S06 algorithm outputs；
- AlphaPose、YOLO-Pose high、WHAM raw outputs；
- EgoHumans pose2d cache；
- 后续新增真实 smoke 输出。

## 必测边界

1. 缺失文件：立即失败；
2. 空 JSON/空 NPZ：立即失败；
3. NaN/Inf/全零数组：立即失败；
4. joint 数量或顺序错误：立即失败；
5. frame id 非单调：立即失败；
6. source/session split 泄漏：立即失败；
7. 缺失 provenance：标记 `adopted_existing`，不得标记 `verified_current_run`；
8. 相同 content hash 或高度相同张量：标记疑似重复，禁止自动作为独立 source；
9. 2D/3D/SMPL space 混用：立即失败；
10. tracklet/person mapping 不一致：保留错误上下文并停止正式矩阵。

## 通过标准

- 所有正式源通过 schema/content/provenance validator；
- inventory 能重算 joint、bone、missing、confidence、tracklet 摘要；
- manifest hash 不依赖绝对路径或 mtime；
- 报告能列出每个 source 的 `included/excluded/pending` 状态和理由。

## 运行方式

E1 脚本完成后，命令和输出路径必须追加到本文件，并在 `results/results.md` 保存审计摘要。E1 只做只读审计，不启动训练。
