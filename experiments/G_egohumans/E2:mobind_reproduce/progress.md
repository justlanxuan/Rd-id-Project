# E2:mobind_reproduce 实时进度日志

## 2026-06-23 16:30
* 完成全部三项官方评测并汇总结果到 `results/results.md`。
* 生成 3 张可视化图表：`retrieval_r_at_k.png`、`localization_accuracy.png`、`sync_metrics.png`。
* 编写一键复现脚本 `scripts/A1_run_full_repro.sh` 与可视化脚本 `scripts/B1_visualize_results.py`。

### 最终指标（stage2 MAE checkpoint）
* **Retrieval**
  * IMU→Video R@1 = 0.8264，Video→IMU R@1 = 0.8368
* **Localization**
  * Person overall = 98.01%（P2/P3 100%，P4 96.96%）
  * Limb (conditioned on correct person) = 89.22%
* **Sync**
  * Person-level MAE = 0.0421s，Acc@0.2 = 0.9925
  * Video-level MAE = 0.0392s，Acc@0.2 = 1.0000

## 2026-06-23 15:15
* 修复 `cache.py` 与 `cache_multi_person.py` 中 numpy int64 导致 JSON 序列化失败的问题（`window_s` / `window_f` 转 int）。
* 成功构建三个官方 cache：
  * `cache_action_5_2`：4,659 个样本
  * `cache_action_multi_5_2`：299 个样本
  * `cache_sync_action_20_5`：1,540 个条目（使用 MoBind 自带 annotations.txt）
* 准备运行 retrieval / localization / sync 评测。

## 2026-06-23 15:00
* `mobind_repro` conda 环境创建完成：Python 3.10 + PyTorch 2.1.0+cu118。
* 解决 NumPy 2.x 与 torch 2.1 不兼容：降级到 numpy 1.26.4。
* 数据路径对齐：创建符号链接 `/data/lyxie/ReID/Data/egohumans/EgoHumans -> /data/.../egohumans`。
* 创建复现实验目录 `checkpoints/EgoHumans/stage2_repro/`，将 `config.yaml` 的 `root_dir` 指向符号链接路径，并软链接到 `best.pt`。

## 2026-06-23 14:45
* 按 HAROS 规范创建 `E2:mobind_reproduce` 实验沙盒。
* 完成 `plan.md` 与 `test/test.md`。
* 更新上层 `experiments/G_egohumans/plan.md`，将 E2 注册为 G_egohumans 的新子目标。
* 准备创建 conda env `mobind_repro` 并安装依赖。
