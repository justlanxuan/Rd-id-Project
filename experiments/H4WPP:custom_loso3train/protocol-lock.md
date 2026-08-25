# H4W++ Custom 三训练 session / 一测试 session协议

## 目的

评估 Hand4Whole++ skeleton 在 Custom 数据上的跨 session 泛化能力。

## 固定协议

- 四个 session：`20260211_171423`、`20260211_171724`、`20260211_172257`、`20260211_172522`。
- 四折 leave-one-session-out：每折 3 个 session 作为训练集，剩余 1 个 session 作为测试集。

| Fold | 训练 sessions | 测试 session |
|---|---|---|
| 1 | `171724,172257,172522` | `171423` |
| 2 | `171423,172257,172522` | `171724` |
| 3 | `171423,171724,172522` | `172257` |
| 4 | `171423,171724,172257` | `172522` |

- 不设置独立验证 session；训练配置显式使用 `best_metric: train_top1`，避免把测试 session 或额外 session 用于模型选择。
- 窗口长度/步长：`24/16`。
- H4W++ skeleton：root-relative 3-D H36M-17，`skeleton_normalize: true`。
- IMU：按 `imu_person_mapping.json` 的 canonical person 顺序配对，`multi_person: true`，每个测试窗口保留 2 个候选。
- 训练：Hybrid model，20 epochs，batch size 64，seed 42，训练集 IMU stats。
- 测试：FrameAcc 和 Group Test（group size 2/4/6/8，100 trials，30-window chunks）。
- 随机 FrameAcc 基线：50%。

## Artifact边界

- 配置：`configs/custom_h4wpp_loso_*.yaml`
- prepared cache：`/data/fzliang/reid-project/custom/preprocessed/h4wpp_w24/loso_*`
- checkpoint：`/data/fzliang/reid-project/custom/artifacts/train/h4wpp_loso3train/`
- predictions/results：`/data/fzliang/reid-project/custom/artifacts/evaluate/h4wpp_loso3train/`
