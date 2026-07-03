# E1 Progress: EgoHumans Pre-Trained Dual-Embedding Transfer

## 当前状态

- [x] E1 formulation / plan 已创建。
- [x] A1: EgoHumans cache 已构建（`cache_action_1.2_0.5`，21,179 windows）。
- [x] A2/A3 source 训练脚本与 configs 已创建。
- [x] A5 zero-shot 评估脚本已创建。
- [x] A6 target fine-tune 脚本已创建。
- [🔄] A3: Source Model-G seed0 完成（Stage2 val ~0.63）。
- [🔄] A3: Source Model-G seeds 42/123/1 已启动（GPU 1/5/7）。
- [🔄] A2: Source Model-L seed0 训练中（GPU 0，Stage2 val ~0.64，early stopping counter 52/100）。
- [ ] A3: Source Model-G seeds 2/3 待启动。
- [ ] A2: Source Model-L seeds 42/123/1/2/3 待启动。
- [ ] A4: 源域融合评估待训练完成后执行。
- [ ] A5: target zero-shot 待训练完成后执行。
- [ ] A6: target fine-tune 待 zero-shot 完成后执行。
- [ ] A8: 结果聚合待所有实验完成后执行。

## 训练状态

| Branch | Seed | GPU | Status | Best Val Acc |
|---|---|---|---|---|
| Global | 0 | — | 完成 | ~0.63 |
| Global | 42 | 1 | Stage1/2 运行中 | — |
| Global | 123 | 5 | Stage1/2 运行中 | — |
| Global | 1 | 7 | Stage1/2 运行中 | — |
| Global | 2 | — | 待启动 | — |
| Global | 3 | — | 待启动 | — |
| Local | 0 | 0 | Stage2 运行中 | ~0.64 |
| Local | 42/123/1/2/3 | — | 待启动 | — |

## 下一步

1. 等待 Local seed0 完成，释放 GPU 0。
2. 启动 Local seeds 42/123/1/2/3 和 Global seeds 2/3。
3. 所有 source 训练完成后运行 zero-shot / fine-tune。
