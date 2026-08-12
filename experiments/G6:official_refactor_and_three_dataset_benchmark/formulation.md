# Formulation：正式重构与三数据集统一基准

## Need

当前仓库同时存在可工作的历史实现和一批未完成的重构改动。公共入口、模块引用、数据预处理实现与文档之间已经出现断裂；dataset、extractor、model 的选择逻辑分散，难以安全扩展到新数据结构或新模型。另一方面，现有实验跨越多个历史协议，无法直接形成一个统一的 source、zero-shot、fine-tune 和 Custom direct LOSO 主表。

因此需要先以已验证的 HEAD 行为作为语义基线，完成接口化、配置化和测试保护下的重构，再冻结代码与实验协议，运行统一基准。工程重构和实验结论必须分开验证，避免把结构性回归误判为模型效果变化。

## Goal

建立一个稳定的兼容层：

```text
Config
  -> Pipeline orchestration
      -> DatasetAdapter registry
      -> Extractor registry
      -> canonical intermediate schema
      -> generic training Dataset
      -> Model registry
      -> Metric registry
      -> canonical result records
```

用户只需运行：

```bash
python run_pipeline.py --config xxx.yaml
python run_pipeline.py --config xxx.yaml --stages preprocess
python run_pipeline.py --config xxx.yaml --stages train
python run_pipeline.py --config xxx.yaml --stages test
python run_pipeline.py --config xxx.yaml --stages preprocess,train,test
```

即可执行默认全流程或任意合法阶段组合。配置决定具体 DatasetAdapter、Extractor、Model 和 Metric 实现；注册表只负责解析与构造，不承载数据或模型业务逻辑。

随后在同一套协议下回答三个问题：

1. TotalCapture 与 EgoHumans 各自的 source-domain FrameAcc 是多少？
2. 两个 source 模型迁移到 Custom 时，zero-shot 与 fine-tune 的 FrameAcc 分别是多少？
3. 不使用 source 预训练、直接在 Custom 上进行 leave-one-session-out 训练时，FrameAcc 是多少？

所有 Custom 问题都必须同时回答 Custom 整体和每个 session 的结果，并在至少三个固定 seed 上重复。

## 研究对象与变量

### 固定对象

- Primary model：当前 `IMUVideoMatcher` hybrid 模型；重构只改变构造和调用边界，不改变数学语义。
- Canonical IMU：单侧 left-wrist / left-lower-arm 的 `acc(3) + quat(4)`，最终映射必须由数据预检确认并写入 manifest。
- Canonical skeleton：H36M-17 语义；原始 GT 或 extractor 输出必须经显式 adapter 转换。
- Window：主协议固定 `window_len=24`、`stride=16`；不得在结果单元之间混用。
- Extraction cache：正式训练允许复用已有骨架；复用前必须验证内容、schema、来源视频和 backend/config provenance。每个数据集另做一次 `force=true` 的独立真实提取 smoke，以证明 extractor 在当前环境可工作。
- Seeds：`[0, 42, 123]`。
- Target outer folds：Custom 四个 session 各作为一次完全 held-out test session。

### 自变量

- source：`totalcapture` / `egohumans` / `none`。
- target adaptation：`zero_shot` / `finetune` / `direct`。
- held-out Custom session：四个 session 之一。
- seed：`0` / `42` / `123`。

### 因变量

- Primary：严格匹配协议下的 FrameAcc。
- Supporting：`correct`、`total`、`weighted_frame_acc`、窗口级均值/标准差、候选组规模、singleton rate。
- Custom aggregate：macro-session FrameAcc 与 micro/weighted FrameAcc。
- Diagnostic：Group Test 结果可保留，但不能替代主 FrameAcc。

## 假设与有效性约束

### H1：接口重构等价性

在相同输入、seed、权重与配置下，重构前后的张量形状、前向输出和指标计算在规定容差内一致。

### H2：迁移实验可比较性

zero-shot、fine-tune 和 direct 在完全相同的 Custom outer folds、输入 schema、窗口与 evaluator 下评估；差异只来自初始化/训练方式。

### H3：Custom 无泄漏

held-out session 不得参与训练、验证、归一化统计、超参数选择、early stopping 或 checkpoint selection。外层测试前生成并校验 split manifest。

### H4：FrameAcc 有判别力

每个 FrameAcc candidate group 原则上应有至少 2 个候选。TotalCapture 单人序列必须通过确定性、可复现的跨序列候选组构造形成多候选匹配；同时报告 group-size 分布和 singleton rate。若无法满足，则该结果标为 invalid，不能以 1.0 填入主表。

### H5：缓存复用不替代提取验证

已有 skeleton artifact 必须先验证内容和 schema。若原 artifact 带 provenance，则同时核对来源视频、extractor 名称和关键配置 hash；历史 artifact 缺少这些信息时允许一次性审计后标为 `provenance_status=adopted_existing`，不得伪称原配置已验证。三个数据集仍必须分别在隔离 smoke 目录中真正运行一次 extractor；smoke 结果记录 `cache_status=extracted`，不能以 `reused` 代替。

## Custom LOSO 内层验证策略

四个 session 按上述固定顺序编号。对 outer fold `i`：

- `test_session = session[i]`；
- `val_session = session[(i + 1) mod 4]`；
- 其余两个 session 为 train；
- zero-shot 不使用 train/val，只读取 source checkpoint 并在 test session 评估；
- fine-tune 与 direct 使用完全相同的 target train/val/test manifest；
- checkpoint 只能依据 val 指标选择。

这是主报告的严格协议。若未来希望利用全部三个非测试 session 训练，必须作为独立附加协议，不能与主结果混表。

## 成功指标

### 工程成功

- CLI 契约、注册表与 schema 全部通过自动测试。
- 三个正式 dataset adapter 和至少当前正式 extractor/model 可由配置创建。
- 历史 checkpoint 的兼容路径有显式测试或显式迁移错误。
- 不再存在“返回空数据但继续运行”的正式实现。

### 实验成功

- 最低运行矩阵完整，而不是只保留最优 seed。
- 每个结果可追溯到 commit、config hash、data manifest、seed、checkpoint 与原始计数。
- 主表完整汇报结果，无论效果好坏。
- 统计脚本可从单次 run records 重新生成所有汇总表。

本 Goal 不预设性能必须超过某个数值；防止以结果导向方式反复修改测试协议。性能结论只能在协议冻结后由数据给出。
