# 📋 E2:mobind_reproduce 实验计划

## 1. 科学目标
复现 **MoBInd 官方预训练模型在 EgoHumans 数据集上的评测指标**，作为 Autism-project Re-ID pipeline 的第三方基线，验证 MoBInd 本身在该数据上的跨模态检索、人员定位与时序同步性能。

## 2. 对照组与实验组

| 子实验 | 变量 | 目的 | 状态 |
|--------|------|------|------|
| **A1** | 环境搭建 | 创建独立 conda env `mobind_repro`，安装 MoBind 依赖 | 待执行 |
| **A2** | 数据路径对齐 | 创建 EgoHumans 目录符号链接/调整 `DATA_ROOT`，使 MoBind cache/eval 脚本能找到 `extracted_data` | 待执行 |
| **A3** | 构建官方 cache | 运行 `cache.py`、`cache_multi_person.py`、`cache_sync.py`，生成 contrastive / multi-person / sync cache | 待执行 |
| **A4** | 官方评测 | 使用 `checkpoints/EgoHumans/stage2` 跑 `eval_retrieval.py`、`eval_localization.py`、`eval_sync_egoh.py` | 待执行 |
| **A5** | 结果整理 | 将 MoBind 指标与 Autism-project 结果横向对比，写入 `results.md` 与可视化 | 待执行 |

## 3. 数据集与资源

* MoBInd 仓库：`/home/fzliang/MoBind`
* 预训练权重：`/home/fzliang/MoBind/checkpoints/EgoHumans/stage1` 与 `stage2`
* EgoHumans 数据：`/data/lyxie/ReID/Data/egohumans/extracted_data/*.npy`
* 目标 cache 目录：`/data/lyxie/ReID/Data/egohumans/cache_action_5_2`、`cache_action_multi_5_2`、`cache_sync_action_20_5`
* GPU：`cuda:4`（RTX 4090 D 24GB）
* Conda env：`mobind_repro`（新创建）

## 4. 评估指标

* **Retrieval**：`eval_retrieval.py` → IMU→Video / Video→IMU R@1/3/5/10/25/50，mean/median rank
* **Localization**：`eval_localization.py --task all` → Person Localization Accuracy（按人数分桶）、Limb Localization Accuracy
* **Sync**：`eval_sync_egoh.py --task person/video` → MAE(s)、Acc@0.1s/0.2s/0.5s

## 5. 预估时耗与资源开销

| 阶段 | 预估时间 | 资源 |
|------|---------|------|
| A1 环境安装 | 10–20 分钟 | 网络 + 磁盘 |
| A2 路径对齐 | < 5 分钟 | CPU |
| A3 cache 构建 | 10–30 分钟 | CPU + 磁盘 I/O |
| A4 评测 | 10–30 分钟 | GPU 4 |
| A5 结果整理 | 20 分钟 | CPU |

## 6. 风险与拦截点

* 若 `torch` 与当前 CUDA 11.8 不兼容，需降级/调整 PyTorch 版本。
* 若 `extracted_data` 中序列数与 MoBind 官方 split 不一致，需核对缺失序列并记录。
* 若 cache 构建或 eval 出现维度错误，需检查 `DATA_ROOT` 对齐与 `extracted_data` 字段完整性。
