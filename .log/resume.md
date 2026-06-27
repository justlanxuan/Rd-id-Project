# 🔄 HAROS Task Resume Node

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E10/E10b 已完成，SOTA 已按 HAROS 要求归档**
* **当前具体执行任务:** 按 HAROS 要求记录 E10/E10b 切分方式下的 SOTA 结果，并更新 resume、progress、results 与 formulation。
* **当前子任务状态:**
  - ✅ E10: 3-seed EgoHumans + custom 联合预训练与评估完成。
  - ✅ E10b: 3-seed 仅 custom 同切分训练与评估完成。
  - ✅ SOTA 条目已写入 `experiments/G_egohumans/formulation.md`、E10/E10b 的 `progress.md` / `results.md`，以及 HAROS 官方看板 `experiments/SOTA_reproduce.md`。

## 2. 最新成果
* **E10b 同切分 custom-only:** custom test clips 平均 FrameAcc **0.613 ± 0.010**（7 clips，seeds 0/42/123）。这是当前 **custom per-video 7:3 split（每个视频先按 ~1800 帧切段，每段再 7:3 切 train/test）** 下的 SOTA。
* **E10 联合预训练:** EgoHumans test clips **0.719 ± 0.011**（24 clips）；custom test clips **0.611 ± 0.078**（7 clips）。
* **关键结论:** E10 与 E10b 在 custom 上无显著差异（0.611 vs 0.613），说明 E10 相对 E9 的提升主要来自切分方式 / 分段策略，而非 EgoHumans 数据本身。
* 结果文件：
  - `experiments/G_egohumans/E10:joint_pretraining/results/results.md`
  - `experiments/G_egohumans/E10b:custom_only_same_split/results/results.md`

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* **备注:** `experiments/` 目录在 `.gitignore` 中，因此 E10/E10b 的实验文档无法进入 Git；SOTA 的代码级锚点只能依赖 tracked 的 `SOTA_reproduce.md`。

## 4. 下一步行动 (Next Actions)
* [ ] 人类决策：下一步是继续探索 EgoHumans → custom 的域自适应，还是基于 E10b SOTA 撰写报告？
