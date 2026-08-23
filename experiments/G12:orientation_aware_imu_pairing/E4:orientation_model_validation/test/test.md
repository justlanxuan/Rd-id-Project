# E4 Test Contract

- baseline 与 turning-gate 必须使用同一 train/eval manifests、candidate groups、normalization、batch sampler、epoch/step 数和 seeds。
- baseline 的 forward 必须完全忽略 orientation；turning-gate 必须消费 `ORIENTATION_SCHEMA` 的五个字段，且 orientation 在源 skeleton bbox normalization 前计算。
- orientation 输出 shape 固定为 `[target_len,5]`，数值 finite；invalid orientation 不得产生 NaN，rate 必须裁剪到 `[-1,1]`。
- 每个 run 必须报告 Custom23/57/22/24 的 raw `correct/total`、FrameAcc、mean margin；turning-gate 另报告 gate mean/std。
- 每个 seed 的 epoch 选择只能依据 Custom23；不能查看 57/22/24 后再选模型。
- 0.8 s 和 2.0 s 必须分开解释；2.0 s 结果不得替代 G11 已冻结的 0.8 s primary protocol。
- physical correlation、gate 激活和匹配提升必须分开表述；不能把单 seed peak 写成稳定收益。
