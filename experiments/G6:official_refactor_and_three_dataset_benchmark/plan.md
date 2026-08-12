# Plan：正式重构与三数据集统一基准

## 0. 执行原则

1. 遵循 HAROS：Formulation → Plan → 测试定义 → 实现/实验 → Progress → Result → 经验沉淀。
2. 重构前先保护当前可验证行为；不把算法改动混入结构重构。
3. 所有正式 GPU 实验必须基于可识别的代码 commit、冻结配置和 split manifest。
4. 任何正式实验必须至少运行 seed `0, 42, 123`；失败 seed 不得静默丢弃。
5. Custom test 只用于最终评估；不得根据 held-out session 结果选择超参数。
6. 大型数据、checkpoint、预测和日志写入 `/data/fzliang/reid-project`；仓库只跟踪代码、配置、manifest 摘要和 Markdown 结果。
7. 每一步同步更新 `.log/progress/YYYY-MM-DD.md`、`.log/resume.md` 和对应实验 `progress.md`。

## 1. 目标架构

### 1.1 公共入口

根目录保留唯一正式入口：

```bash
python run_pipeline.py --config CONFIG [--stages STAGES]
```

- 未提供 `--stages`：依次执行 `preprocess,train,test`。
- `--stages` 接受逗号分隔的有序子集。
- `prepare` 可作为 `preprocess` 的弃用别名，`evaluate` 可作为 `test` 的弃用别名；输出 warning，不静默改变行为。
- 每个 stage 在运行前验证其输入 artifact；缺失时给出可执行的错误信息。
- `preprocess` 内部可细分 extract、adapt、pack、validate，但这些是内部步骤，不增加用户必须理解的顶层命令。
- 骨架 extraction 采用显式缓存策略：`reuse_existing=true` 时先验证缓存；`force=true` 时强制提取；无效缓存默认报错，只有配置 `invalid_cache_policy=reextract` 才重提取。

### 1.2 接口边界

```text
Raw dataset
  -> DatasetAdapter
  -> canonical sequence records (NPZ + CSV/manifest)
  -> Extractor (when raw video requires it)
  -> canonical skeleton/IMU records
  -> WindowAlignmentDataset
  -> Model
  -> ModelOutput
  -> Metric
  -> RunRecord / AggregateRecord
```

正式抽象：

- `DatasetAdapter`：TotalCapture、EgoHumans、Custom 各自负责原始目录发现、时间同步、坐标/关节/传感器映射和 canonical record 生成。
- `Extractor`：统一 `prepare/check_dependencies/extract` 契约；正式实现至少覆盖 AlphaPoseFull 和 ByteTrack+AlphaPose，WHAM 明确标注 experimental。
- `WindowAlignmentDataset`：只消费 canonical schema，不了解具体原始数据集。
- `ReIDModel`：统一输入、`ModelOutput`、capabilities 和 checkpoint metadata；首个正式注册项为当前 hybrid matcher。
- `Metric`：FrameAcc 和 GroupTest 采用注册表，主实验只认一个冻结版本的 FrameAcc protocol id。

避免建立一个同时处理 dataset、extractor、model 的“万能工厂”。每个 domain 独立 registry，非法名称立即失败，并输出可用实现列表。

### 1.3 Canonical schema

schema 至少包含：

- `schema_version`
- `dataset`, `session`, `sequence`, `person_id`
- `timestamps` 或可验证的 `fps/start/end`
- `imu`, `imu_channels`, `imu_location`, `imu_frame`
- `skeleton`, `joint_names`, `skeleton_space`, `confidence`
- `source_sequence`, `candidate_group_id`, `segment_id`
- `split`, `fold_id`
- preprocessing parameters and source hashes

schema validator 必须检查维度、有限值、时间单调性、关节顺序、IMU channel 顺序、人与模态对齐和 split 泄漏。

## 2. 工作分解

### E1：重构契约与行为基线

目标：把可信 HEAD 行为、当前工作树问题和公共契约固化成测试。

任务：

1. 保存当前 dirty worktree 清单，不覆盖人类已有修改。
2. 从可信 HEAD 提取以下 golden/smoke 行为：配置解析、核心 import、dummy forward、checkpoint key、FrameAcc 小样例。
3. 为根 CLI 写 contract tests：默认阶段、独立阶段、组合阶段、错误阶段、artifact 缺失。
4. 为无效 registry key、未完成 extractor、空数据、旧 checkpoint 写 fail-loud tests。
5. 记录重构前后允许的数值容差；默认 deterministic 单元测试要求 exact 或 `atol <= 1e-6`。

完成门：测试先能揭示当前工作树断裂，并可在可信 HEAD 基线通过对应行为测试。

### E2：核心重构与数据兼容层

目标：完成 CLI、配置、接口、registries、schema、三数据集 adapter 和模型兼容层。

建议实施顺序：

1. 修复/恢复公共 import 边界，消除对已删除 `src.preprocess.structures`、`src.modules.domain` 等路径的悬空引用。
2. 建立 typed config schema，并保留现有 YAML 的显式迁移器；禁止未使用字段被悄悄忽略。
3. 实现 stage runner 和 artifact contract。
4. 实现 canonical records、validator、manifest 和 generic `WindowAlignmentDataset`。
5. 依次迁移 TotalCaptureAdapter、EgoHumansAdapter、CustomAdapter；每迁移一个就执行其 toy/real smoke tests。
6. 迁移 AlphaPoseFull、ByteTrackAlphaPose extractor；对 WHAM、Custom+ 和 stub 使用明确 capability/experimental 错误。
   - 删除任何“backend 导入失败就生成空 JSON”的 fallback。
   - 为提取 artifact 保存来源视频、backend、关键配置 hash、生成时间和 cache status。
   - 缓存目录存在不等于可复用；必须校验 JSON 非空、关节结构和有限值。历史缓存缺少 provenance 时记录为 `adopted_existing`，不能伪称原 backend/config 已验证。
7. 实现 ModelRegistry、hybrid model adapter、`ModelOutput` 和 checkpoint metadata/legacy loader。
8. 实现 MetricRegistry，冻结 FrameAcc protocol id 和记录 schema。
9. 清理重复 factory、过期脚本、文档和配置引用；保留必要的 deprecated shim，并给出移除期限。
10. 更新 README、示例 config 和中文“重构代码” skill。

完成门：

- `compileall`、全部单元/集成测试通过；
- 所有正式 YAML 可加载并 dry-run；
- 三个 dataset 的真实小样本 preprocess 通过 schema validator；
- hybrid dummy forward、单步 train、save/load、test 通过；
- 根 CLI 的五种要求用法均通过；
- 无 silent fallback、TODO 正式路径或 stale import。

### E3：统一数据预检与协议冻结

目标：在任何大规模 GPU 训练前，证明三个数据集进入模型的语义一致且 split 无泄漏。

任务：

1. 生成三个 dataset manifests，统计 session、sequence、person、时长、窗口、候选组和缺失率。
2. 确认 left-wrist/left-lower-arm 的实际字段映射；保存单位、坐标系、采样率和 quaternion 顺序。
3. 确认 H36M-17 joint mapping，禁止仅凭张量长度推断语义。
4. 固定窗口 `24/16`、归一化方法和 FrameAcc protocol id。
5. 检查每个 fold 的 train/val/test `session` 与 `source_sequence` 完全不相交。
6. 对 TotalCapture 构建确定性的多候选 group，确保主结果 singleton rate 为 0；group 构造算法、K 和随机种子写入 manifest。
7. 对每个 dataset 运行 1 个 epoch / 少量 batch 的 seed-0 smoke，不进入最终统计。
8. 对三个数据集分别执行一次真实 extractor smoke，不允许命中缓存：
   - TotalCapture：一个短视频，`force=true`，独立 smoke 输出目录；
   - EgoHumans：一个短视频，`force=true`，独立 smoke 输出目录；
   - Custom：一个短视频，`force=true`，独立 smoke 输出目录；
   - 每次保存 backend、命令/config hash、运行环境、输出骨架摘要和 `cache_status=extracted`。
9. 正式 preprocess/train 可切换到已验证的既有骨架缓存，生成 reuse manifest，避免重复提取完整数据集。
10. 生成人类可审核的 `protocol-lock.md`；审核后配置只读冻结。协议变更必须新建 protocol version，并使旧/新结果不可混合。
11. 经人类授权创建实验代码 snapshot commit，将其写入 protocol hash；正式调度器在 dirty worktree 或 HEAD 与锁定 commit 不一致时必须拒绝执行。非正式 smoke 不进入主结果。

完成门：所有数据 preflight tests 通过，protocol hash 固定并绑定干净代码 commit，人在正式 GPU 扩展前确认协议。

### E4：Source-domain 训练与评估

目标：分别训练 TotalCapture、EgoHumans，形成 source checkpoint 与 source FrameAcc。

运行矩阵：

```text
source ∈ {totalcapture, egohumans}
seed   ∈ {0, 42, 123}
```

每个 run 输出：

- resolved config、protocol hash、commit、environment；
- best/last checkpoint 及选择依据；
- source test predictions；
- FrameAcc、correct、total、weighted FrameAcc；
- candidate group-size distribution 与 singleton rate；
- training/validation curves 和 wall-clock/resource summary。

完成门：6 个 source train/test 单元完整，并生成 source 汇总表。若 TotalCapture singleton rate 非 0，结果不能进入主表。

### E5：Source -> Custom Zero-shot

目标：不使用任何 Custom 训练/验证数据，直接评估每个 source seed checkpoint 在四个 Custom test sessions 上的迁移表现。

运行矩阵：

```text
source  ∈ {totalcapture, egohumans}
session ∈ {171423, 171724, 172257, 172522}
seed    ∈ {0, 42, 123}  # 对应 source training seed
```

要求：

- source normalization 只能来自 source train；
- 不得用 Custom session 估计 normalization 或选择 checkpoint；
- 24 个评估结果全部保留；
- 同时输出逐 session、macro-session 和 micro/weighted 汇总。

### E6：Source -> Custom Fine-tune

目标：使用 source checkpoint 初始化，在严格 Custom LOSO 协议下 fine-tune。

运行矩阵：

```text
source       ∈ {totalcapture, egohumans}
outer_fold   ∈ {4 custom sessions}
seed         ∈ {0, 42, 123}
initial_ckpt = checkpoint(source, seed)
```

要求：

- 24 个训练/评估单元；
- target train/val/test manifest 与 direct 条件完全一致；
- seed 同时控制 source checkpoint 对应关系和 target fine-tune 随机性；
- 所有超参数在第一个正式 test 结果暴露前冻结；
- checkpoint 仅由 inner validation session 选择。

### E7：Custom Direct LOSO

目标：不加载 source 权重，从随机初始化在 Custom 上训练。

运行矩阵：

```text
outer_fold ∈ {4 custom sessions}
seed       ∈ {0, 42, 123}
```

要求：12 个训练/评估单元；数据、模型、训练预算和 evaluator 与 fine-tune 一致，唯一主差异是初始化方式。

### E8：汇总、复现与验收

目标：从机器可读 run records 自动生成最终报告，不手抄数值。

任务：

1. validator 检查所有 required cells、seed、hash、artifact 是否齐全。
2. 重新计算每个 run 的 FrameAcc，并和保存值逐项核对。
3. 生成逐 seed、逐 session、逐 condition 和整体结果表。
4. 计算 seed `mean ± sample std`；Custom 同时计算 macro-session 与 micro/weighted。
5. 对 fine-tune vs direct、fine-tune vs zero-shot 提供配对 seed/fold 差值；至少报告效应量和 bootstrap 95% CI，统计推断标注为探索性（只有 3 seeds）。
6. 在干净环境复现随机选定的至少一个 source、一个 zero-shot、一个 fine-tune 和一个 direct 单元。
7. 完成代码审查、文档、HAROS progress/result/retrospective 和中文 skill 更新；经人类确认后提交。

## 3. 结果记录 schema

每次评估保存一条不可变记录，至少包括：

```yaml
run_id: ...
protocol_id: ...
protocol_hash: ...
git_commit: ...
dataset: totalcapture|egohumans|custom
source: totalcapture|egohumans|none
adaptation: source|zero_shot|finetune|direct
outer_test_session: null|...
inner_val_session: null|...
seed: 0|42|123
config_hash: ...
data_manifest_hash: ...
checkpoint: ...
frame_acc:
  correct: ...
  total: ...
  value: ...
  weighted_value: ...
candidate_groups:
  min_size: ...
  mean_size: ...
  singleton_rate: ...
status: completed|failed|invalid
```

预测和中间相似度矩阵保存到 artifact 路径；Markdown 只引用摘要和相对 artifact id。

## 4. 最终结果表模板

### 4.1 Source-domain

| Source | Seed | FrameAcc | Correct / Total | Weighted FrameAcc | Singleton rate |
|---|---:|---:|---:|---:|---:|
| TotalCapture | 0 | TBD | TBD | TBD | TBD |
| TotalCapture | 42 | TBD | TBD | TBD | TBD |
| TotalCapture | 123 | TBD | TBD | TBD | TBD |
| TotalCapture | mean ± std | TBD | — | TBD | TBD |
| EgoHumans | 0 | TBD | TBD | TBD | TBD |
| EgoHumans | 42 | TBD | TBD | TBD | TBD |
| EgoHumans | 123 | TBD | TBD | TBD | TBD |
| EgoHumans | mean ± std | TBD | — | TBD | TBD |

### 4.2 Custom 逐 session

| Adaptation | Source | Test session | Seed 0 | Seed 42 | Seed 123 | Mean ± std | Correct / Total |
|---|---|---|---:|---:|---:|---:|---:|
| zero-shot | TotalCapture | 20260211_171423 | TBD | TBD | TBD | TBD | TBD |
| zero-shot | TotalCapture | 20260211_171724 | TBD | TBD | TBD | TBD | TBD |
| zero-shot | TotalCapture | 20260211_172257 | TBD | TBD | TBD | TBD | TBD |
| zero-shot | TotalCapture | 20260211_172522 | TBD | TBD | TBD | TBD | TBD |
| zero-shot | EgoHumans | each session | TBD | TBD | TBD | TBD | TBD |
| fine-tune | TotalCapture | each session | TBD | TBD | TBD | TBD | TBD |
| fine-tune | EgoHumans | each session | TBD | TBD | TBD | TBD | TBD |
| direct | none | each session | TBD | TBD | TBD | TBD | TBD |

最终生成器必须展开所有 `each session`，模板中的简写不能出现在最终报告。

### 4.3 Custom 整体

| Adaptation | Source | Macro session FrameAcc | Micro/weighted FrameAcc | Seed std | Session std |
|---|---|---:|---:|---:|---:|
| zero-shot | TotalCapture | TBD | TBD | TBD | TBD |
| zero-shot | EgoHumans | TBD | TBD | TBD | TBD |
| fine-tune | TotalCapture | TBD | TBD | TBD | TBD |
| fine-tune | EgoHumans | TBD | TBD | TBD | TBD |
| direct | none | TBD | TBD | TBD | TBD |

## 5. 测试与验收清单

### 静态与单元测试

- [x] Python compile/import。
- [x] 全部配置 schema validation。
- [x] 每个 registry 的创建、重复注册、未知 key、capability mismatch。
- [x] canonical schema 正例和缺字段/错维度/NaN/时间逆序反例。
- [x] FrameAcc 手工可验小矩阵，包括 singleton 拒绝或明确计数行为。
- [x] model dummy forward、loss、backward、checkpoint round-trip。
- [x] legacy checkpoint compatibility。

### 集成测试

- [x] 五种 CLI 用法。
- [ ] preprocess、train、test 可跨进程独立衔接。
- [x] 三个 dataset toy fixture 全流程。
- [x] 三个 dataset 真实小样本 smoke。
- [x] 三个 dataset 各一次强制真实骨架提取 smoke，且缓存复用/无效缓存/显式重提取测试通过。
- [x] split leakage validator 主动注入泄漏时失败。
- [x] 汇总器缺 seed、重复 run、hash 不一致、失败 run 时失败。

### 实验验收

- [x] protocol lock 在正式训练前完成。
- [ ] 6 source runs。
- [ ] 24 zero-shot evaluations。
- [ ] 24 fine-tune runs。
- [ ] 12 direct runs。
- [ ] 所有逐 session 与整体表自动生成。
- [ ] 任意抽样结果可从 artifact 重算。

## 6. 运行调度与资源控制

1. 先对每条实验路径运行单 fold、单 seed smoke，测量峰值显存、单 epoch 时间和 artifact 体积。
2. 根据 smoke 结果生成运行清单和预计 GPU-hours，不事先假设固定耗时。
3. GPU 为共享资源；每次调度前检查空闲显存，避免占满所有设备。并发度由实测峰值和当时资源决定。
4. 所有任务使用可恢复的 job manifest；中断后只补缺失单元，不重复覆盖成功 artifact。
5. 每个 run 写独立目录并采用原子完成标记；汇总器只读取 `completed` 且 hash 匹配的记录。
6. `/home` 空间紧张，重型 artifact 一律写入 `/data/fzliang/reid-project`，定期记录磁盘占用。

## 7. 决策门与回滚

| Gate | 进入条件 | 失败处理 |
|---|---|---|
| G-A 重构实现 | contract tests 已定义 | 修复测试/基线，不启动数据迁移 |
| G-B 数据 smoke | 核心重构测试通过 | 回滚到最近可验证小步，禁止 silent workaround |
| G-C 正式训练 | schema、split、protocol-lock 全通过且人类确认 | 只修协议/数据，旧结果标 invalid，不混用 |
| G-D 结果汇总 | required run matrix 全部完成 | 同 seed 重跑失败单元；无法补齐则明确 incomplete |
| G-E 最终提交 | 复现、文档、skill、人类审核完成 | 保留工作树，不擅自 commit |

## 8. 交付物

- 重构后的代码、测试和迁移文档。
- 根 `run_pipeline.py` 与三数据集正式配置。
- protocol lock、四个 Custom fold manifests 和全部 resolved configs。
- source/zero-shot/fine-tune/direct checkpoint 与 run records。
- 自动汇总脚本和机器可读 CSV/JSON。
- 最终中文报告：逐 source、逐 session、整体 macro/micro、至少 3 seeds。
- HAROS formulation/plan/progress/results/retrospective。
- 根据本轮全部反馈更新并验证的中文“重构代码” skill。

## 9. 暂不启动项

在本计划得到人类确认且 E1/E2 测试门未通过前，不启动 42 个正式训练任务。现有历史结果只用于行为核对和背景，不直接填入新协议主表。
