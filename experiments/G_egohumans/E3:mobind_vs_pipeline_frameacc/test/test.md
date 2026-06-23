# 🧪 E3:mobind_vs_pipeline_frameacc 测试说明

## 测试目标
验证在相同 16 个 EgoHumans 序列上，MoBInd 官方 checkpoint 与我们训练的 pipeline 都能成功输出 FrameAcc；并比较三种聚合方式（单窗口、4 窗口 embedding 平均、4 窗口 assignment 投票）。

## 测试项

| 子实验 | 对象 | 通过标准 |
|--------|------|----------|
| A1 | `scripts/A1_eval_mobind_frameacc.py` | 输出 `results/mobind_frameacc.json` |
| A2 | `scripts/A2_eval_ours_frameacc_subset.py` | 输出 `results/ours_frameacc.json` |
| A4 | `scripts/A4_eval_ours_frameacc_4window.py` | 输出 `results/ours_frameacc_4window.json` |
| A5 | `scripts/A5_eval_ours_frameacc_4window_vote.py` | 输出 `results/ours_frameacc_4window_vote.json` |
| A3 | `scripts/A3_compare_results.py` | 生成包含三种聚合方式的对比表格、图表和 `results/results.md` |

## 关键约束
- 测试序列排除 MoBInd official test/val：移除 `01_002`, `03_001`, `05_002`, `04_005`。
- MoBInd 使用 5 秒窗口（100 帧）。
- 我们的模型使用 24 帧窗口；A4/A5 每 4 个连续窗口聚合后做一次决策。

## 失败判定
- A1 因张量形状不匹配报错 → 检查 MoBInd 输入格式。
- A2/A4/A5 序列数不是 16 → 检查 CSV 过滤逻辑。
- `--group_vote` 无效 → 检查 `eval_synchronous.py` 是否正确处理投票分支。
