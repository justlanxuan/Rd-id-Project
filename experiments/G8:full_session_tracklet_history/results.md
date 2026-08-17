# G8 初步结果

评测日期：2026-08-17。窗口 24，步长 16，完整 session，原始未合并
tracklet。历史规则仅在推理启用：逐 tracklet hard-threshold SignedVote，
`decay=0`、`sigmoid(3 × margin) > 0.7` 才更新，否则 preserve。

## Source 预训练模型（seed 0）

| Source | Session | Tracklets | 历史 FrameAcc | 即时 FrameAcc |
|---|---|---:|---:|---:|
| EgoHumans | 20260211_171423 | 14 | 43.22% | 58.77% |
| EgoHumans | 20260211_171724 | 2 | 0.00% | 29.53% |
| EgoHumans | 20260211_172257 | 2 | 66.05% | 37.84% |
| EgoHumans | 20260211_172522 | 3 | 40.99% | 79.42% |
| EgoHumans | **micro** | — | **36.65% (5625/15349)** | **51.50%** |
| TotalCapture | 20260211_171423 | 14 | 44.11% | 59.34% |
| TotalCapture | 20260211_171724 | 2 | 100.00% | 99.30% |
| TotalCapture | 20260211_172257 | 2 | 81.98% | 50.45% |
| TotalCapture | 20260211_172522 | 3 | 24.72% | 40.42% |
| TotalCapture | **micro** | — | **62.58% (9606/15349)** | **62.95%** |

## E28 Custom LOSO `best.pt`（seed 42）

每个 session 只使用未在该 session 上训练的对应 fold checkpoint。

| Session | Tracklets | 历史 FrameAcc | 即时 FrameAcc |
|---|---:|---:|---:|
| 20260211_171423 | 14 | 90.09% (3527/3915) | 50.75% |
| 20260211_171724 | 2 | 95.76% (3840/4010) | 71.02% |
| 20260211_172257 | 2 | 94.31% (3350/3552) | 52.87% |
| 20260211_172522 | 3 | 0.83% (32/3872) | 16.89% |
| **macro-session** | — | **70.25%** | **47.88%** |
| **micro** | — | **70.03% (10749/15349)** | **48.00%** |

前三个 session 显示历史累积可显著稳定预测；第四个 session 是明确反例，
早期错误会被 `decay=0` 的累计状态锁死。不得用前三折结果掩盖该失败，也
不得在看到第四折测试结果后选择 session-specific 参数并作为正式结果。

原始逐窗口相似度、历史分数、初始化/更新/冻结事件及逐帧分配保存在：

```text
/data/fzliang/reid-project/custom/evaluation/full_session_tracklet_history/
```

## 三 seed 稳定性检查

评测日期：2026-08-17。使用 G6 中独立训练的 seed 0、42、123
checkpoint；窗口、步长、完整 session、未合并 tracklet 和 SignedVote 参数
均与上文相同。这里的 seed 是模型训练 seed，不是对同一 checkpoint 重复
设置评测随机种子。E28 只有 seed 42，因此不参与三 seed 统计。

### Source 模型迁移到 Custom

| Source | Seed | 历史 micro | 即时 micro | 历史 - 即时 |
|---|---:|---:|---:|---:|
| EgoHumans | 0 | 36.65% | 51.50% | -14.85 pp |
| EgoHumans | 42 | 38.29% | 61.28% | -22.99 pp |
| EgoHumans | 123 | 50.09% | 72.28% | -22.18 pp |
| EgoHumans | **mean ± sample std** | **41.68 ± 7.34%** | **61.68 ± 10.40%** | **-20.01 pp** |
| TotalCapture | 0 | 62.58% | 62.95% | -0.36 pp |
| TotalCapture | 42 | 27.20% | 38.17% | -10.96 pp |
| TotalCapture | 123 | 53.53% | 62.28% | -8.76 pp |
| TotalCapture | **mean ± sample std** | **47.77 ± 18.38%** | **54.47 ± 14.12%** | **-6.70 pp** |

在两个 source 上，三个 seed 的历史 micro 都不高于各自即时基线，方向一致；
但绝对结果和退化幅度随训练 seed 大幅变化，尤其 TotalCapture 的历史结果
从 27.20% 到 62.58%。

### Custom direct LOSO

| Seed | 171423 | 171724 | 172257 | 172522 | micro | 即时 micro |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 44.96% | 0.00% | 100.00% | 59.01% | 49.50% | 40.04% |
| 42 | 71.80% | 0.00% | 100.00% | 59.01% | 56.34% | 34.36% |
| 123 | 84.24% | 0.00% | 0.00% | 35.72% | 30.50% | 35.79% |
| **mean ± sample std** | **67.00 ± 20.08%** | **0.00 ± 0.00%** | **66.67 ± 57.74%** | **51.25 ± 13.45%** | **45.44 ± 13.39%** | **36.73 ± 2.96%** |

历史决策的平均 micro 比即时基线高 8.72 pp，但这个均值不代表稳定收益：
seed 0 和 42 分别提高 9.45、21.98 pp，seed 123 反而降低 5.29 pp。
172257 更在 100%、100%、0% 之间翻转；171724 则三个 seed 均为 0%。

因此，当前 `decay=0` 的 hard-threshold SignedVote **不能视为 seed 稳定的改进**。
它会将模型早期的配对方向长期累积：初始方向正确时接近满分，方向错误时
接近零分。跨 seed 的数值波动明显大于即时匹配。E28 seed 42 的 70.03%
只能作为单 seed 结果，不能外推为稳定结论。

三 seed 原始预测保存在：

```text
/data/fzliang/reid-project/custom/evaluation/full_session_tracklet_history/multiseed_g6/
```
