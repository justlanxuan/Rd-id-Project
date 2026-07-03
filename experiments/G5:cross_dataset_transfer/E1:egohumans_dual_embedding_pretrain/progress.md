# E1 Progress: EgoHumans Pre-Trained Dual-Embedding Transfer

## 当前状态

- [x] E1 formulation / plan 已创建。
- [x] A1: EgoHumans cache 已构建（`cache_action_1.2_0.5`，21,179 windows）。
- [x] A2/A3 source 训练脚本与 configs 已创建。
- [x] A5 zero-shot 评估脚本已创建。
- [x] A6 target fine-tune 脚本已创建。
- [x] A3: Source Model-G seeds 0/1 完成；seeds 42/123 运行中。
- [x] A2: Source Model-L seed0 完成；seeds 1/42 运行中。
- [ ] A3: Source Model-G seeds 2/3 待启动。
- [ ] A2: Source Model-L seeds 123/2/3 待启动。
- [ ] A4: 源域融合评估待训练完成后执行。
- [x] A5: target zero-shot **seed0 已完成**：Mean FrameAcc = 0.2940。
- [x] A6: target fine-tune **Local seed0 已完成**：Mean FrameAcc = **0.7259**。
- [x] A6: target fine-tune **Global seed0 已完成**。
- [x] A6: target fine-tune **Fusion seed0 已完成**：Mean FrameAcc = **0.7332**（best α=0.9）。
- [ ] A8: 结果聚合待所有实验完成后执行。

## 关键发现

### Seed0 Zero-Shot
- Mean FrameAcc = **0.2940**（pure local best）。
- Global branch 有害。

### Seed0 Fine-Tune
| 设置 | Best α | Mean FrameAcc |
|---|---|---|
| Local only | 1.0 | 0.7259 |
| Local + Global fusion | 0.9 | **0.7332** |

- 较 zero-shot 提升 **+0.439**。
- 接近 G4/E11 from-scratch（0.752 ± 0.095）。
- Global branch 在目标域 fine-tune 后变得有用。

## 训练状态

| Branch | Seed | GPU | Status | Best Val Acc |
|---|---|---|---|---|
| Global | 0/1 | — | 完成 | ~0.63 |
| Global | 42 | 1 | Stage1/2 运行中 | — |
| Global | 123 | 5 | Stage1/2 运行中 | — |
| Global | 2/3 | — | 待启动 | — |
| Local | 0 | — | 完成 | ~0.64 |
| Local | 1 | 7 | 运行中 | — |
| Local | 42 | 0 | 运行中 | — |
| Local | 123/2/3 | — | 待启动 | — |
| Local fine-tune | 0 | — | 完成 | 0.7259 |
| Global fine-tune | 0 | — | 完成 | — |
| Fusion fine-tune | 0 | — | 完成 | 0.7332 |

## 下一步

1. 等待 source seeds 42/123/1（local/global）完成。
2. seed1 local 完成后，立即跑 seed1 fine-tune。
3. 继续扩展至 6 seeds，统计 mean/std 与 from-scratch 对比。
