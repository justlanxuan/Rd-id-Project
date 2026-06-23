# 🧪 E3:mobind_vs_pipeline_frameacc 测试说明

## 测试目标
验证在相同 16 个 EgoHumans 序列上，MoBInd 官方 checkpoint 与我们训练的 pipeline 都能成功输出 FrameAcc，并且二者可比；进一步验证 4 窗口聚合对我们 pipeline 的影响。

## 测试项

| 子实验 | 对象 | 通过标准 |
|--------|------|----------|
| A1 | `scripts/A1_eval_mobind_frameacc.py` | 输出 `results/mobind_frameacc.json`，包含 16 个序列的 per-sequence FrameAcc 和 mean FrameAcc |
| A2 | `scripts/A2_eval_ours_frameacc_subset.py` | 过滤 CSV 后运行 `eval_synchronous.py`，输出 `results/ours_frameacc.json` |
| A4 | `scripts/A4_eval_ours_frameacc_4window.py` | 使用 `--group_windows 4` 运行 `eval_synchronous.py`，输出 `results/ours_frameacc_4window.json` |
| A3 | `scripts/A3_compare_results.py` | 生成包含 1-window 与 4-window 的对比表格、图表和 `results/results.md` |

## 关键约束
- 测试序列必须排除 MoBInd official test/val：
  - 移除 `01_002`, `03_001`, `05_002`（在 MoBInd test 中）
  - 移除 `04_005`（在 MoBInd val 中）
- MoBInd 模型使用 5 秒窗口（100 帧），因为 `ConvFormer` 固定该长度。
- 我们的模型使用训练时的 24 帧窗口；A4 中每 4 个连续窗口聚合后做一次决策。

## 失败判定
- A1 因张量形状不匹配报错 → 检查 IMU/motion 输入是否与 MoBInd `ContrastiveMAE.forward` 一致。
- A1/A2/A4 输出的序列数不是 16 → 检查序列过滤逻辑。
- A2/A4 出现 CUDA OOM → 降低 `--batch_size`。
- `--group_windows 4` 无效果 → 检查 `src/engine/eval_synchronous.py` 是否正确传入该参数。
