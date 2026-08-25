# Progress：H4W++ Custom 三训练 session四折实验

## 2026-08-25

### 已完成

- 固定 leave-one-session-out 协议：每折 3 个 session 训练、1 个 session 测试，不使用独立验证 session。
- 为无验证集训练补充 prepared-cache 校验兼容：仅当 `slice.val_sessions` 明确为空时允许空 `windows_val.csv`；默认有验证配置仍拒绝空 split。
- 创建四份独立配置、四个隔离 prepared 目录和对应 HAROS 协议锁定文件。
- 四折窗口已生成并通过窗口长度/步长及双候选测试组检查：训练窗口分别为 714、734、762、736；测试窗口分别为 268、248、220、246。

### 运行中

- 正在依次执行四折 `preprocess -> train -> test`，GPU 使用物理卡 1。
- 训练 checkpoint 选择指标固定为 `train_top1`，每折训练 20 epochs。

### 待补充

- 每折训练曲线、checkpoint、FrameAcc 原始 `correct/total`、Group Test 结果。
- 四折 macro/micro 汇总、随机基线比较及异常审计。

## 2026-08-25（完成）

### 四折结果

| 测试 session | 训练窗口 | 测试窗口 | FrameAcc | Group-2 | Group-4 | Group-6 | Group-8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20260211_171423` | 714 | 268 | `160/268=59.70%` | 57.0% | 48.5% | 35.0% | 25.0% |
| `20260211_171724` | 734 | 248 | `228/248=91.94%` | 75.0% | 32.5% | 16.83% | 12.5% |
| `20260211_172257` | 762 | 220 | `194/220=88.18%` | 92.0% | 78.0% | 64.0% | 50.0% |
| `20260211_172522` | 736 | 246 | `206/246=83.74%` | 100.0% | 92.5% | 83.0% | 62.5% |

- FrameAcc 宏平均：`80.89%`；宏标准差（总体）：`12.57%`。
- 所有测试窗口加权：`788/982=80.24%`，随机基线为 50%。
- Group Test 宏平均：group-2 `81.00%`、group-4 `62.88%`、group-6 `49.71%`、group-8 `37.50%`。
- 四折均训练至 20 epochs，未使用测试 session 做模型选择；checkpoint 选择指标为训练集 batch retrieval `train_top1`。
- 代码/协议验证：prepared cache 允许空 val 仅在 `val_sessions` 明确为空时生效；其余默认空 split 仍失败。`tests/test_adapter_validation.py`：12 passed；compileall 与 `git diff --check` 通过。
- 仓库入口真实 H4W++ 单帧 smoke 通过：使用 pinned-compatible adapter、H4W++ snapshot、WiLoR、DWPose 和 SMPL-X/FLAME/MANO 资产，在 2160 帧视频上按 `frame_stride=100000` 产生 2 个有效 17×3 skeleton detections；输出字段和有限性检查通过。当前 smoke 复用了本机许可资产，资产本身不进入 Git。

### Artifact

- 训练：`/data/fzliang/reid-project/custom/artifacts/train/h4wpp_loso3train/`
- 测试：`/data/fzliang/reid-project/custom/artifacts/evaluate/h4wpp_loso3train/`
- 每折 `epoch_metrics.jsonl`、`metrics.json`、`best.pt`、`last.pt` 和完整 `results.json` 均已保存。
