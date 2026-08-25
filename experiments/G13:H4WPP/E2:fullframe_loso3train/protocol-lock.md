# G13 E2：H4W++ 全帧推理 / 三训练 session 一测试 session

## 目的

在 G13 E1 稀疏 H4W++ 基线之上，移除稀疏推理：对每个原始视频帧运行一次
Hand4Whole++，并使用全帧 skeleton 生成训练和测试窗口。

## 唯一变量

| 项目 | E1 稀疏基线 | E2 全帧对照 |
|---|---:|---:|
| H4W++ inference frame stride | 16 | 1 |
| 中间帧填充 | 有 | 无（仅处理实际检测缺失时保留必要的时间对齐逻辑） |
| window length | 24 | 24 |
| window stride | 16 | 16 |
| session split | 相同四折 LOSO | 相同四折 LOSO |
| model / epochs / seed | hybrid / 20 / 42 | 相同 |

## 固定协议

- Sessions：`20260211_171423`、`20260211_171724`、`20260211_172257`、`20260211_172522`。
- 每折三个 session 训练，剩余一个 session 测试；不使用独立验证集。
- `best_metric: train_top1`，避免测试集参与 checkpoint 选择。
- 每个窗口保留两个人候选，使用 canonical IMU person mapping。
- 评估：FrameAcc 和 Group Test（group size 2/4/6/8，100 trials）。
- E2 只改变 H4W++ inference frame stride；不得切换模型架构或数据划分。

## 复现命令

全帧 skeleton 生成命令必须显式包含 `--frame-stride 1`。生成四个 fold 的 prepared
cache 后，使用 G13 E2 配置运行：

```bash
python tools/run_h4wpp_loso.py \
  --config-prefix custom_h4wpp_fullframe_loso \
  --gpu 0 --stages preprocess,train,test
```

实际输出目录、配置 hash、checkpoint hash 和结果 hash 写入本实验的
`artifact-manifest.json`。
