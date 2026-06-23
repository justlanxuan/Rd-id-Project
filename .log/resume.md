# 🔄 HAROS Task Resume Node

> ⚠️ **写给 AI:** 本文件是人机协作的唯一状态机断点。在每次迭代结束或实验中断前，必须精确更新以下字段，严禁留空。

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E2:mobind_reproduce 已完成**
* **当前具体执行任务:** MoBInd 官方 EgoHumans 基线复现（retrieval / localization / sync）已全部跑通并归档。
* **当前子任务:** 完成 E2 结果汇总与文档更新。

## 2. 最新成果
* E2 实验沙盒已按 HAROS 规范创建并完结：
  * `experiments/G_egohumans/E2:mobind_reproduce/results/results.md`
  * `experiments/G_egohumans/E2:mobind_reproduce/progress.md`
  * `experiments/G_egohumans/E2:mobind_reproduce/scripts/A1_run_full_repro.sh`
  * `experiments/G_egohumans/E2:mobind_reproduce/scripts/B1_visualize_results.py`
  * `results/figures/` 下 3 张图表
* 复现结果（stage2 MAE checkpoint）：
  * Retrieval：IMU→Video R@1 = 0.8264，Video→IMU R@1 = 0.8368
  * Localization：Person 98.01%，Limb 89.22%
  * Sync：Person-level MAE 0.0421s / Acc@0.2 0.9925；Video-level MAE 0.0392s / Acc@0.2 1.0000
* MoBind 本地修改未提交到 upstream（外部依赖仓库）：
  * `preprocess/EgoHumans/cache.py`
  * `preprocess/EgoHumans/cache_multi_person.py`
  * `builder/build_model.py`
  * `eval_sync_egoh.py`

## 3. 当前阻塞痛点 (Blockers & Issues)
* MoBInd 为外部仓库，本地修复未 commit/push，仅用于当前复现。
* `experiments/` 目录在 Autism-project `.gitignore` 中，E2 文档未进 git（HAROS 实验区本地化）。

## 4. 下一步行动 (Next Actions)
* [ ] 决定是否需要将 E2 实验区文件强制 add 到 `egohumans` 分支。
* [ ] 决定是否需要将 MoBind 修改 fork/ patch 保存。
* [ ] 基于 E2 基线，设计后续改进实验（E3），例如：
  * 使用自定义 pose estimator / 2D keypoint 提取替换官方输入；
  * 在 sync 任务上引入 TCN/Transformer 对齐改进；
  * 对比单模态 vs 跨模态检索在 EgoHumans 上的性能边界。
