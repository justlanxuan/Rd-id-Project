# E1 Plan: EgoHumans Pre-Trained Dual-Embedding Transfer

## 实验设计

| 子实验 | 名称 | 设置 | 目的 |
|---|---|---|---|
| A1 | 数据准备 | 构建/确认 EgoHumans local + global cache | 确保源域训练数据可用 |
| A2 | 源域训练 Model-L | EgoHumans 上训练 local branch（wjoint, RightWrist） | 获得源域 local checkpoint |
| A3 | 源域训练 Model-G | EgoHumans 上训练 global branch（pose2d） | 获得源域 global checkpoint |
| A4 | 源域融合评估 | EgoHumans test 上 α sweep | 验证源域 dual-embedding 有效性 |
| A5 | target zero-shot | 加载 A2/A3 checkpoint，直接评估 custom test | 测试跨域泛化性 |
| A6 | target full fine-tune | 加载 A2/A3 checkpoint，在 custom 上 full fine-tune | 测试域适应能力 |
| A7 | target partial fine-tune | 冻结 Stage1，只 fine-tune Stage2 | 测试保留源域 encoder 的效果 |
| A8 | 结果聚合 | 汇总 source / zero-shot / fine-tune 结果 | 与 G4/E11 对比 |

## 数据路径

- **源域数据（EgoHumans）：**
  - local: `data/interim/egohumans_mobind_aligned_single_imu_24/`（或 w24 对应 cache）
  - global: 需要确认是否存在 pose2d cache，若不存在需新建。
- **目标域数据（custom）：**
  - `experiments/G3:custom_failure_diagnosis/E2:mobind_on_custom_same_split/data/`

## 检查点约定

- 源域 checkpoint 保存路径：
  - `experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain/artifacts/source_w24_seed{SEED}_local/`
  - `experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain/artifacts/source_w24_seed{SEED}_global/`
- 目标域 fine-tune checkpoint 保存路径：
  - `experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain/artifacts/target_w24_seed{SEED}_local/`
  - `experiments/G5:cross_dataset_transfer/E1:egohumans_dual_embedding_pretrain/artifacts/target_w24_seed{SEED}_global/`

## 对照基线

- G4/E11 custom from-scratch w24 Fusion best α: **0.752 ± 0.095**
- G_egohumans/E9 E8 → custom zero-shot: **0.339**

## 风险与应对

- **风险 1:** EgoHumans pose2d cache 不存在。应对：复用 E2/A1b 或 E11 的 cache 构建逻辑，为 EgoHumans 生成 pose2d cache。
- **风险 2:** 源域训练耗时。应对：先在 1 seed 上验证数据 pipeline，再扩展到多 seed。
- **风险 3:** Fine-tune 破坏源域表示。应对：A7 partial fine-tune 作为 fallback。
