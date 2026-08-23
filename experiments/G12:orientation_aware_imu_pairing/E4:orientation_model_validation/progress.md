# E4 Progress

- [x] 建立 `OrientationMotionDataset`：从 extractor skeleton 在 bbox normalization 前派生可审计 turning stream。
- [x] 建立 `OrientationAwareMatcher`：baseline、learned turning gate、concat/gyro-focus/residual controls。
- [x] 完成 0.8 s baseline/turning-gate 三 seed screen。
- [x] 完成 2.0 s baseline/turning-gate 三 seed screen。
- [x] 运行 dataset smoke、py_compile、Ruff focused validation。
- [x] 生成 E4 结果、测试契约与结论。

## Artifact

模型和 metrics 位于 `/data/fzliang/reid-project/g12/e4_orientation_model/`。每个 run 保存 `metrics.json`, `best.pt`, `last.pt`；训练脚本写入完整 config 和 orientation contract。
