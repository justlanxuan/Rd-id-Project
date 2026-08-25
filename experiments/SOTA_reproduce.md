# Re-id-Project 当前结果锚点

## H4W++ Custom LOSO SOTA（2026-08-25）

在 Hand4Whole++ 3-D H36M-17 skeleton、`24/16` 窗口、严格三训练 session/一测试
session LOSO 协议下，当前 Custom session-matching SOTA 为：

- FrameAcc 宏平均：`80.89%`
- 全部测试窗口加权：`788/982 = 80.24%`
- 四个测试 session：`59.70% / 91.94% / 88.18% / 83.74%`
- 2-way 随机基线：`50%`

该结果已经获得人类授权记录为 SOTA，并绑定到本仓库的 clean release commit；完整
协议、训练曲线、原始 predictions 和 SHA-256 manifest 见
[`experiments/H4WPP:custom_loso3train/`](H4WPP:custom_loso3train/)。

模型选择不使用测试 session 或独立验证 session，20 epochs 后按训练集
`train_top1` 保存 best checkpoint。该选择契约与有验证集的历史四折结果不同，必须
按本节协议复现。

## 已知基线看板

| 条件 | 指标 | 数值 | 来源 |
|---|---|---:|---|
| EgoHumans → Custom zero-shot | micro FrameAcc | 62.02% | `experiments/G9:source_to_custom_gap/formulation.md` |
| EgoHumans → Custom fine-tune | micro FrameAcc | 34.46% | 同上 |
| TotalCapture → Custom zero-shot | micro FrameAcc | 54.06% | 同上 |
| TotalCapture → Custom fine-tune | micro FrameAcc | 50.92% | 同上 |

这些数值是 G6/G9 formulation 中引用的历史基线，不代表已完成 G9 因果归因。

## 代码与协议锚点

- 发布分支：`main`
- SOTA release commit：见本文件提交后的 Git history。
- 正式 artifact 根目录：`/data/fzliang/reid-project/`
- 推荐入口：`./run_pipeline.py --config CONFIG.yaml`

## 复现门禁

- `git clone --recurse-submodules` 后执行 `python tools/setup_h4wpp.py --install`；
- 通过 `python tools/setup_h4wpp.py --check` 验证模型资产；
- 使用 `configs/custom_h4wpp_loso_*.yaml` 顺序执行 `preprocess,train,test`；
- 用 `artifact-manifest.json` 复核 config、CSV、checkpoint 和 raw prediction hash；
- 外部 SMPL/FLAME/MANO 和 H4W++ checkpoint 必须由用户按上游许可证提供，不能从
  Git 历史猜测或替换成空文件。
