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

- 每个 source 都有 `included/conditional/excluded/pending` 决策和理由；`included` 子集通过 schema/content/provenance validator；
- inventory 能重算 joint、bone、missing、confidence、tracklet 摘要；
- manifest hash 不依赖绝对路径或 mtime；
- 报告能列出每个 source 的状态和理由；只要 source、Custom target、person/IMU/time join 的最小可信子集存在，就允许继续 gap 分析。

## 运行方式

当前审计命令：

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \\
  experiments/G9:source_to_custom_gap/E1:gap-audit/scripts/A1_build_source_inventory.py \\
  --sample-limit 4 --full-hash --max-npz-inspect-mb 64
```

输出：`/data/fzliang/reid-project/g9/e1_gap_audit/source_inventory.json`。

语义审计命令：

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \\
  experiments/G9:source_to_custom_gap/E1:gap-audit/scripts/A2_semantic_skeleton_audit.py
```

输出：`/data/fzliang/reid-project/g9/e1_gap_audit/semantic_audit.json`，其中包含全量 Custom fold CSV 映射、S06 baseline 的 person/IMU join 和最小可信子集。

Gap manifest 命令（只基于已审计 JSON，不训练）：

```bash
/home/fzliang/miniconda3/envs/reid_project/bin/python \\
  experiments/G9:source_to_custom_gap/E1:gap-audit/scripts/A3_build_gap_profile.py
```

输出：`/data/fzliang/reid-project/g9/e1_gap_audit/gap_profile.json`。

本命令只读取已有 artifact；大型 NPZ 超过阈值时只记录文件信息，不解压到内存。E1 最终通过前，还必须追加全量/多序列 fingerprint、逐关节 outlier 和 source/Custom coverage 检查。E1 只做只读审计，不启动训练。
