# E1 Progress: EgoHumans Pre-Trained Dual-Embedding Transfer

## 当前状态

- [x] E1 formulation / plan 已创建。
- [x] A1: EgoHumans cache 已构建（`cache_action_1.2_0.5`，21,179 windows，兼容 custom w24 架构：24 frames, 4 patches, patch_size=6）。
- [x] A2/A3 source 训练脚本与 configs 已创建。
- [x] A5 zero-shot 评估脚本已创建。
- [x] A6 target fine-tune 脚本已创建。
- [🔄] A2: Source Model-L seed0 训练中（GPU 0）。
- [🔄] A3: Source Model-G seed0 训练中（GPU 1）。
- [ ] A4: 源域融合评估待训练完成后执行。
- [ ] A5: target zero-shot 待训练完成后执行。
- [ ] A6: target fine-tune 待 zero-shot 完成后执行。
- [ ] A8: 结果聚合待所有实验完成后执行。

## 训练状态

- `source_w24_seed0_local`: Stage1 运行中，约 epoch 27，val R@1 ~0.18。
- `source_w24_seed0_global`: Stage1 刚启动。

## 下一步

1. 等待 seed0 两个 branch 训练完成，验证 pipeline 无错误。
2. 若 seed0 成功，启动全部 6 seeds 的 source 训练。
3. 运行 A5 zero-shot 评估。
4. 运行 A6 fine-tune 评估。
