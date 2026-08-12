# G7 Stride-24 Ablation Protocol Lock

状态：`locked`

人类请求：2026-08-12（Asia/Hong_Kong），“如果步长为 24，结果如何？”

## 唯一自变量

- `window_len=24`、`stride=24`，训练、验证、测试和 Custom segment evaluator 全部统一。
- 模型、50 epoch 预算、seed `0/42/123`、数据语义、Custom 循环 inner validation、
  source/zero-shot/fine-tune/direct 条件及 FrameAcc 定义保持 G6 不变。
- 这是独立消融；不得覆盖或混入 G6 的正式 artifact。

## 基线口径勘误

审计真实 CSV 后确认：G6 的 TotalCapture/EgoHumans prepared cache 为 stride 16；
Custom prepared cache 实际为 stride 8（尽管 G6 resolved config 写 16）；G6 Custom
segment evaluator 为 stride 16。因此 G7 与 G6 的 Custom 训练侧比较实际是 24 vs 8，
评估侧是 24 vs 16，不能描述为严格的单一 16→24 对照。

## 数据与 folds

- TotalCapture：同一 canonical sequence tensors，按 subject split，以 24 重新切窗。
- EgoHumans：同一 canonical sequence tensors，按 session split，以 24 重新切窗。
- Custom：同一 segment skeleton、raw CSV 7D IMU、person-order 修正与四折循环
  inner validation，以 24 重新切窗。
- 六个 manifest 的 split identity 和 `source_sequence` 均无跨 split 泄漏；manifest
  从 CSV 推导的实际 stride 均为 24。

| Dataset/fold | Train/val/test rows | Manifest hash |
|---|---:|---|
| TotalCapture | 5937/716/586 | `63197e7b2bad62bccc0891400133297699a0da2e30bccf502ac287abe18844d1` |
| EgoHumans | 873/192/744 | `00fe440f82b5e09a145558df969d5466a569482e90c79b92738d4f84e0646284` |
| Custom fold 1 | 310/164/145 | `00d608ab6847f115651e1a57abb79a8aef42a30b2d74c75faeab06b82e4ed3d1` |
| Custom fold 2 | 307/148/164 | `33e2cc9c1f87e5107bae457be90ffde4582703b9934dd15178bd8ff2ca58e7d2` |
| Custom fold 3 | 309/162/148 | `43528f0c00f0f6bd44c8576c5f90a0f7185083227506aed3916f0a8ceb2f0ea3` |
| Custom fold 4 | 312/145/162 | `3f8f668c8f66585391faed1c9319f023ea52195ec50ead3d3f3e1e9aa757852e` |

## 正式矩阵

- 与 G6 相同：42 training + 66 evaluation = 108 jobs。
- 每项三个 seed；保留逐 seed `correct/total`、逐 Custom session、macro/micro 与
  sample standard deviation。
- protocol hash 绑定新的 clean snapshot commit、`stride24` profile、配置、环境、
  manifest 与 required-cell 矩阵。
