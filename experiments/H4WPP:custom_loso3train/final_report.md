# H4W++ Custom 三训练 session四折结果

## 结论

在每折使用三个 session 训练、剩余 session 测试的严格 LOSO 协议下，H4W++ skeleton + Hybrid IMU–skeleton 模型取得 `80.89%` 的 FrameAcc 宏平均，所有测试窗口加权结果为 `80.24%`。四个测试 session 均为双候选组，随机基线为 `50%`。

## FrameAcc

| 测试 session | 正确/总数 | FrameAcc |
|---|---:|---:|
| `20260211_171423` | 160/268 | 59.70% |
| `20260211_171724` | 228/248 | 91.94% |
| `20260211_172257` | 194/220 | 88.18% |
| `20260211_172522` | 206/246 | 83.74% |
| 宏平均 | — | 80.89% |
| 加权总体 | 788/982 | 80.24% |

## Group Test

四折宏平均准确率为：group size 2：`81.00%`；4：`62.88%`；6：`49.71%`；8：`37.50%`。

## 与前一轮四折的关系

前一轮采用的是 `2 train + 1 val + 1 test`，宏平均为 `66.48%`。本轮不使用独立验证 session，而是用三个 session 全量训练，因此训练数据更多；但 checkpoint 选择改为显式 `train_top1`，不使用测试 session 进行选择。两种结果反映的是不同训练协议，不能仅归因于 H4W++ skeleton 质量变化。

## 可复现文件

- 协议：`protocol-lock.md`
- 配置：`../../configs/custom_h4wpp_loso_*.yaml`
- checkpoint：`/data/fzliang/reid-project/custom/artifacts/train/h4wpp_loso3train/`
- 完整评估：`/data/fzliang/reid-project/custom/artifacts/evaluate/h4wpp_loso3train/`

## 仓库内兼容层

- H4W++ 官方源码固定为 submodule commit `f81d35d`。
- WiLoR 固定为 `fcb9113`，MMPose 固定为 `71ec36e`；由
  `tools/setup_h4wpp.py` 自动初始化并建立相对路径链接。
- `third-party/patches/wilor_img_feat.patch` 暴露 H4W++ hand-control 所需的
  `img_feat` 接口。
- `environment-h4wpp.yml` 声明提取环境；SMPL/SMPL-X/MANO/FLAME 和模型 checkpoint
  只通过 `--weights-root` 或官方许可下载提供，不伪造、不提交大文件。
- 已通过仓库脚本真实单帧 smoke，确认不需要旧的独立 Re-id/Hand4Whole 工作区路径。
