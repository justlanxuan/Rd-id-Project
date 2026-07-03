# E1 Progress: EgoHumans Pre-Trained Dual-Embedding Transfer

## 当前状态

- [x] E1 formulation / plan 已创建。
- [x] A1: EgoHumans cache 已构建（`cache_action_1.2_0.5`，21,179 windows）。
- [x] A2/A3 source 训练脚本与 configs 已创建。
- [x] A5 zero-shot 评估脚本已创建。
- [x] A6 target fine-tune 脚本已创建。
- [x] A3: Source Model-G seeds 0/1 完成；seeds 42/123 运行中。
- [x] A2: Source Model-L seed0 完成；seed42 运行中。
- [ ] A3: Source Model-G seeds 2/3 待启动。
- [ ] A2: Source Model-L seeds 123/1/2/3 待启动。
- [ ] A4: 源域融合评估待训练完成后执行。
- [x] A5: target zero-shot **seed0 已完成**：Mean FrameAcc = 0.2940。
- [x] A6: target fine-tune **Local seed0 已完成**：Mean FrameAcc = **0.7259**（best α=1.0）。
- [🔄] A6: target fine-tune **Global seed0 运行中**（GPU 7）。
- [ ] A8: 结果聚合待所有实验完成后执行。

## 关键发现

### Seed0 Zero-Shot
- Mean FrameAcc = **0.2940**（pure local best）。
- Global branch 有害，α 越大性能越好。

### Seed0 Local Fine-Tune
- Mean FrameAcc = **0.7259**（pure local best）。
- 较 zero-shot 提升 **+0.432**。
- 已接近 G4/E11 from-scratch（0.752 ± 0.095）。
- `171724_seg0/seg1` 从 zero-shot 完全失败变为完美（1.0）。

## 训练状态

| Branch | Seed | GPU | Status | Best Val Acc |
|---|---|---|---|---|
| Global | 0/1 | — | 完成 | ~0.63 |
| Global | 42 | 1 | Stage1/2 运行中 | — |
| Global | 123 | 5 | Stage1/2 运行中 | — |
| Global | 2/3 | — | 待启动 | — |
| Local | 0 | — | 完成 | ~0.64 |
| Local | 42 | 0 | 运行中 | — |
| Local | 123/1/2/3 | — | 待启动 | — |
| Local fine-tune | 0 | — | 完成 | 0.7259 FrameAcc |
| Global fine-tune | 0 | 7 | 运行中 | — |

## 下一步

1. 等待 Global fine-tune seed0 完成，评估 dual fine-tuned fusion。
2. 继续启动剩余 source seeds。
3. 若 Global fine-tune 也有收益，扩展 multi-seed fine-tune。
