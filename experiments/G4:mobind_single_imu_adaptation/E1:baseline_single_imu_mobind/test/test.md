# E1 Test Plan

## 测试目标

确保 G4/E1 基线复现与评估流程可一键运行、结果可复现。

## 测试对象

- `scripts/eval_all_seeds.sh`（待创建）
- `scripts/A4_aggregate.py` 的 6-seed 聚合
- 已有的 G3/E2 训练产物与结果文件

## 通过标准

1. 运行评估脚本后，`results/multi_seed_summary.json` 中 w24/w100 的 6-seed 结果与已跑出的数值一致（允许浮点误差 < 1e-4）。
2. `results/results.md` 中的表格与 JSON 完全一致。
3. 脚本在单 GPU 上能在 10 分钟内完成 6 个 seed 的评估。

## 边界条件

- 如果某个 seed 的 stage2 checkpoint 缺失，脚本应报错并跳过，而不是生成错误结果。
- 如果 `MoBind/configs/config.py` 的 Custom `limb_list` 被改回 5 肢体，评估应因 `--limb_list` 参数而仍然正确。

## 复现步骤

```bash
cd /home/fzliang/Autism-project/experiments/G4:mobind_single_imu_adaptation/E1:baseline_single_imu_mobind
bash scripts/eval_all_seeds.sh
python ../../G3:custom_failure_diagnosis/E2:mobind_on_custom_same_split/scripts/A4_aggregate.py \
  --seeds 0 42 123 1 2 3 \
  --out_json multi_seed_summary.json
```

## 失败判定

- 汇总 FrameAcc 与 G3/E2 已产出结果偏差 > 0.001。
- 脚本运行时间 > 30 分钟（不含训练）。
