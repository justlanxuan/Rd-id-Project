# G13 E3：H4W++ 推理密度梯度 3-seed 结果

## 结论

在 `10 inference strides × 3 seeds × 4 LOSO folds = 120` 个独立训练/测试 run 中，
数值最高且 seed 波动最小的是全帧推理 `stride=1`：四折 FrameAcc 的 3-seed 平均为
`73.80% ± 0.41%`。因此当前 profiling 的推荐密度是每个视频帧运行一次 H4W++。

`stride=16` 的 3-seed 平均为 `72.84% ± 5.10%`，只比 stride 1 低 `0.96` 个百分点，
但 seed 波动显著更大。按照预注册决策规则，不能宣称 stride 1 对 stride 16 存在稳定
显著优势；更准确的结论是：stride 1 提供最高均值和最强稳定性，而 stride 16 在
seed 42 上可以获得很高的单次结果，但不稳定。

## 3-seed 四折宏平均

| inference stride | 约推理频率 | seed 0 | seed 42 | seed 123 | 3-seed mean ± population std | 加权 mean |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | 30.000 Hz | 74.30% | 73.79% | 73.29% | **73.80% ± 0.41%** | **72.91%** |
| 2 | 15.000 Hz | 69.66% | 71.22% | 73.71% | 71.53% ± 1.67% | 70.67% |
| 4 | 7.500 Hz | 68.48% | 80.97% | 66.80% | 72.08% ± 6.32% | 71.15% |
| 8 | 3.750 Hz | 66.16% | 77.62% | 68.55% | 70.78% ± 4.93% | 69.93% |
| 12 | 2.500 Hz | 68.55% | 75.21% | 56.75% | 66.84% ± 7.63% | 66.12% |
| 16 | 1.875 Hz | 70.36% | 79.94% | 68.22% | 72.84% ± 5.10% | 72.03% |
| 24 | 1.250 Hz | 63.27% | 68.36% | 59.62% | 63.75% ± 3.59% | 62.73% |
| 32 | 0.938 Hz | 58.35% | 70.60% | 63.47% | 64.14% ± 5.02% | 63.20% |
| 48 | 0.625 Hz | 54.48% | 69.86% | 58.34% | 60.89% ± 6.53% | 60.01% |
| 64 | 0.469 Hz | 58.75% | 65.84% | 58.69% | 61.10% ± 3.36% | 60.29% |

每个 seed 数值先对四个 held-out session 做等权宏平均；表中 `3-seed mean` 再对
seeds `0/42/123` 等权平均。加权 mean 则使用每个 seed 的全部 `982` 个测试 assignment。

## 对 E1/E2 的解释

- E2 的 stride 1 / seed 42 结果为 `73.79%`，在 E3 中被精确复现。
- E3 stride 16 / seed 42 为 `79.94%`，接近但不等于旧 E1 的 `80.89%`。E3 所有密度
  都从同一份全帧 H4W++ JSON 确定性下采样；旧 E1 是单独运行的稀疏 H4W++，存在
  最大约 `0.003–0.008` 的 GPU 浮点差异。
- 旧 E1 `80.89%` 是 seed 42 单次结果。E3 证明该区域 seed 敏感：stride 16 的
  seed 0/42/123 分别为 `70.36% / 79.94% / 68.22%`。
- 当 stride 大于窗口长度 24 后，许多窗口只包含一个独立推理姿态，均值下降到约
  `61–64%`。这支持骨架时序信息仍然有用，而不是越稀疏越好。

## Artifact

- Prepared cache：`/data/fzliang/reid-project/custom/preprocessed/h4wpp_density_w24/`
- 120 个 run：`/data/fzliang/reid-project/custom/artifacts/h4wpp_density_sweep/runs/`
- 汇总：同目录下 `summary.json` 与 `seed_summary.csv`
- 完整 config/checkpoint/results SHA-256：同目录下 `artifact-manifest.json`
- 一键入口：`tools/run_h4wpp_density_sweep.py`

Artifact 总大小约 `2.9 GB`，prepared cache 约 `101 MB`。所有 120 个 run 都存在
`results.json`，不存在缺失 cell。
