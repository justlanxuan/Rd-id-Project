# G6：正式重构与三数据集统一基准

## 目标

按照 HAROS 完成 Re-id-Project 的生产级重构，并在版本冻结、协议统一、无数据泄漏的前提下，完成 TotalCapture、EgoHumans、Custom 三个数据集的训练及迁移到 Custom 的统一实验。

## 范围

### 工程范围

- 保留根入口 `run_pipeline.py`。
- 默认运行 `preprocess -> train -> test`，同时支持阶段独立或组合运行。
- 建立可配置的 DatasetAdapter、Extractor、Model、Metric 接口和注册表。
- 统一中间数据 schema、配置校验、checkpoint 格式、结果 schema 与错误语义。
- 迁移现有 TotalCapture、EgoHumans、Custom 能力；Custom+ 和 WHAM 可保留为显式 experimental，但不得伪装成已完成能力。
- 重构期算法行为冻结；任何算法改动必须独立立项，不能混入结构重构。

### 实验范围

- source 数据集：TotalCapture、EgoHumans。
- target 数据集：Custom。
- 三个分别训练的数据集：TotalCapture、EgoHumans、Custom。
- Custom 外层测试 session：
  - `20260211_171423`
  - `20260211_171724`
  - `20260211_172257`
  - `20260211_172522`
- 主实验 seed：`0, 42, 123`；若增加 seed，只能增加，不能替换或挑选性删除。

## 必交实验结果

1. TotalCapture source test FrameAcc：逐 seed 与 `mean ± std`。
2. EgoHumans source test FrameAcc：逐 seed 与 `mean ± std`。
3. TotalCapture -> Custom zero-shot：四个 session 分别及 Custom 整体结果。
4. EgoHumans -> Custom zero-shot：四个 session 分别及 Custom 整体结果。
5. TotalCapture -> Custom fine-tune：四个 LOSO fold 分别及 Custom 整体结果。
6. EgoHumans -> Custom fine-tune：四个 LOSO fold 分别及 Custom 整体结果。
7. Custom direct training：四个 LOSO fold 分别及 Custom 整体结果。

每个结果必须同时保存原始计数、逐 seed 值、均值和标准差。Custom 整体必须同时给出：

- `macro_session_frame_acc`：四个 session 等权平均；
- `micro/weighted_frame_acc`：累计 `correct / total`；
- session 间离散程度；
- seed 间离散程度。

## 最低运行矩阵

| 类别 | 训练数 | Source test | Custom 评估数 | 说明 |
|---|---:|---:|---:|---|
| source training | 2 sources × 3 seeds = 6 | 6 | 0 | 每个 source 独立训练并测 source test |
| zero-shot | 0 个新增训练 | 0 | 2 sources × 4 sessions × 3 source seeds = 24 | 直接使用 source checkpoint |
| fine-tune | 2 sources × 4 folds × 3 seeds = 24 | 0 | 24 | held-out session 只用于 test |
| Custom direct | 4 folds × 3 seeds = 12 | 0 | 12 | 从随机初始化训练 |
| 合计 | 42 | 6 | 60 | 总评估输出 66；不含 smoke test 和失败重跑 |

## 完成判据

只有同时满足以下条件，G6 才能标记完成：

- [ ] 正式 CLI、接口、schema、配置、checkpoint 与结果汇总全部实现并有测试。
- [ ] 仓库不存在引用已删除模块的正式路径，不存在静默空实现或静默 fallback。
- [ ] 三个数据集均通过最小真实数据 smoke test。
- [ ] TotalCapture、EgoHumans、Custom 各完成至少一次不复用缓存的真实骨架提取 smoke；正式训练可使用通过 schema/provenance 校验的既有提取结果。
- [ ] 42 个最低训练任务及对应评估全部完成；失败任务使用相同 seed 重跑或在最终报告中明确标记缺失，不能直接删除。
- [ ] 所有 required cells 都有配置快照、commit、seed、数据 split manifest、checkpoint 和原始预测/计数记录。
- [ ] Custom held-out session 不出现在 train、validation、normalization statistics 或 checkpoint selection 中。
- [ ] TotalCapture 的 FrameAcc 不是 singleton 候选导致的伪 1.0；最终报告包含候选集规模和 singleton rate。
- [ ] 最终报告包含逐 source、逐迁移方式、逐 Custom session、整体 macro/micro 的完整表格。
- [ ] 从一个干净环境可按文档重现任意一个结果单元。
- [ ] 经人类审核确认后才提交最终代码和更新后的中文“重构代码” skill。

## 非目标

- 不在本 Goal 内声称某个新模型达到 SOTA。
- 不以性能提升作为重构是否成功的替代标准。
- 不把未完成的 Custom+、WHAM 或 detector stub 纳入正式支持列表。
- 不在看到 Custom test 结果后调参并回写主协议。
