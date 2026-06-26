# E7:mobind_full_setting_reproduce 实时进度日志

## 2026-06-26 03:40
* Stage1 训练完成（4h 22m，early stopping），最终 val R@1 ≈ 0.836，与官方 checkpoint 的 retrieval 水平接近。
* 已更新 Stage2 config 的 `stage1_exp` 指向 Stage1 输出目录。
* 开始 A3：Stage2 训练。

## 2026-06-25 23:15
* 修复：官方 YAML 的 `data.root_dir` 是占位符，创建 E7 专用 config，将 `root_dir` 指向真实路径。
* 开始 A2：用 E7 config 重新训练 Stage1。

## 2026-06-25 23:00
* Plan 已审批。
* 恢复 MoBInd `configs/config.py` 的 EgoHumans `limb_list` 为官方默认 5 IMU。
* 初始化 E7 沙盒目录。
* 第一次 Stage1 启动失败：cache 路径因 `root_dir` 占位符未解析。
