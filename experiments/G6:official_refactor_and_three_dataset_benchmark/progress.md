# Progress：G6 正式重构与三数据集统一基准

## 2026-08-11

### 已完成

- 完成代码与资源初步审计，确认可信 HEAD 与当前未完成工作树的差异。
- 与人类对齐根 CLI、DatasetAdapter/Extractor/Model 可配置接口方向。
- 建立正式 Goal、Formulation 和完整 Plan。
- 固定最低实验矩阵、三个主 seed 和四个 Custom outer LOSO sessions。
- 将协议冻结、结果矩阵完整性和逐 session 汇报规则写入中文重构 skill，并通过校验。
- 完成 E1 第一批契约测试与核心断链修复：根 CLI、核心 imports、所有官方配置、模型 CPU forward 和 legacy checkpoint key migration 均有验证。
- 建立独立 DatasetAdapter、Extractor、Model registries；移除按数据集建立空训练 Dataset 子类的方向。
- FrameAcc 增加 singleton 默认拒绝、候选规模统计和跨序列 `candidate_group_id`。
- adapter canonical 输出 validator 拒绝空/全零/错位数据。
- extraction 删除静默空 JSON fallback，增加有效缓存校验、显式重提取策略和 cache/provenance status。
- 人类补充确认：正式训练可复用已有骨架；TotalCapture、EgoHumans、Custom 各需一次强制真实提取 smoke。已同步 Goal、Formulation、Plan 和中文 skill。
- 当前自动验证：`35 passed`，本轮文件 Ruff clean，23 个正式 YAML 加载，核心 import、CLI help、`git diff --check` 通过。

### 当前状态

- E1 契约基线基本完成，E2 正在进行。
- 尚未启动真实 extractor smoke 或正式 GPU 实验。
- EgoHumans/Custom root preprocess 的真实语义仍未恢复，validator 会主动阻止占位数据进入训练。

### 下一步

1. 人类确认计划中的统一窗口与 inner-validation 策略。
2. 恢复 EgoHumans/Custom preprocess 的真实数据路径并通过 adapter validator。
3. 拆分剩余万能 `PipelineFactory`，补齐 MetricRegistry 与 artifact contracts。
4. 为三个数据集建立并预检强制 extraction smoke 配置。

## 2026-08-12

### 已完成

- TotalCapture 正式 G6 preprocess 经公共 CLI 完成：46 个 sequence、11165 个窗口；测试集 879 行全部进入多候选组，IMU 明确为左前臂 `acc3+quat4` 七维语义。
- EgoHumans root adapter 从真实逐人 IMU 与多人 skeleton 生成 canonical 数据：30 个 sequence、2691 个窗口，无 singleton 候选组。
- Custom 四个 LOSO prepared cache 全部通过公共 CLI 的 schema、NPZ、finite、split/session 泄漏与测试候选组检查；singleton 窗口按配置显式排除并记录比例。
- 骨架缓存复用契约区分 `reused/extracted/reextracted` 和 `adopted_existing/verified_current_run`，无效缓存默认失败或显式重提取。
- 删除混合职责的 `PipelineFactory`；根 `run_pipeline.py` 只保留参数与顺序，三个公共 stage 由独立 workflow registry 构造。
- 建立 MetricRegistry；FrameAcc、Group Test 使用统一 EmbeddingBundle，segment FrameAcc 增加逐 session 的 `correct/total`、加权准确率和 clip 平均。
- 完成三个数据集各一次隔离、强制的真实 `alphapose_full` 提取：TotalCapture 20 条、EgoHumans 119 条、Custom 40 条有效姿态；全部为 `verified_current_run`，详情见 `extractor_smoke.md`。
- checkpoint 迁移、legacy stats 与 shape filtering 已移入模型领域；新 checkpoint 写 schema/model/capabilities，train/evaluate 不再按 Hybrid 名称分支。
- 现存 E28 Hybrid checkpoint 真实加载为 `missing=0, unexpected=0, dropped=0`，CPU forward `(2,128)` 且 finite。
- 建立正式 required-cell 矩阵与 validator：42 个 train、66 个 evaluate，覆盖
  source/zero-shot/fine-tune/direct、四个 Custom sessions 与 seeds `0/42/123`。
- 建立不可变 evaluation run record 和完整性汇总器：检查 protocol/config/data/checkpoint
  hash、原始 `correct/total`、逐 session 与 macro/micro 结果。
- 审计 Custom 历史缓存：旧 `hybrid_w24_session_out` 含全零 IMU 窗口且
  skeleton 语义不一致；G6 已统一切换到 raw CSV 7D 且修正两个 session
  person-order 的 `hybrid_w24_session_out_rawcsv7d_swapsess`。
- 六个字节指纹 data manifests 已生成；TotalCapture 按 subject、EgoHumans/Custom
  按 session 检查 split identity，同时检查 `source_sequence`，无泄漏、NaN 或全零 artifact。
- 建立 protocol lock hash 工具与 108 份 resolved-config 生成器；临时目录中全量生成、
  加载并校验了所有依赖 checkpoint、data hash、job id 和运行阶段。
- 修复 YACS 对纯数字式 session 标识的 literal-eval，防止 `20260211_171423`
  被静默转成丢失下划线的整数。
- 实现依赖感知的 G6 可恢复调度器：要求显式 GPU 列表，dry-run 中初始
  18 个无依赖 train 可运行、90 个 job 等待依赖；仅内容/hash 验证完整的
  checkpoint/run record 可被断点跳过，半写 artifact 默认拒绝覆盖。
- 实现独立的 source + Custom 单 epoch smoke config 生成器，可限制每 epoch
  batch 数，使用独立 artifact root 且不生成正式 run record。
- 将 README/development/benchmark/refactor plan 从已删 `src.pipeline/src.preprocess`
  全部迁移到唯一根 CLI 和当前领域 registries；旧 benchmark runner 改为显式 G6
  迁移提示。
- preprocess 直接入口与根 pipeline 现在共用唯一 YACS config loader；未知
  top-level section 失败，旧字段迁移会警告，Custom+ 改为可导入但明确
  experimental 失败。
- 清理生产路径遗留 lint，并移除 extractor dispatcher 的运行时 `sys.path`
  注入与 AlphaPose pose estimator 对外部项目结构的隐藏 fallback；组件配置片段
  使用独立、显式的 fragment parser，并有契约测试保护。
- 对 TotalCapture、EgoHumans 与 Custom 四折分别执行真实训练 batch 的 CPU
  forward/backward：输入、128D embedding、loss 与 54 组梯度张量均 finite；六份
  train split 的 Hybrid fitted stats 全量遍历通过。
- 修正 checkpoint 选择记录：Hybrid 的 `auto` 实际选择 `val_loss`，现在同时保存
  metric 名、原始值和 maximize-oriented score，不再将负 loss 错标为 `best_val_top1`。
- protocol record 现在绑定干净 Git snapshot commit；正式 scheduler 非 dry-run 时
  拒绝 dirty worktree 或 HEAD mismatch，防止只有 `git_dirty=true` 却无法复现实验代码。
- `mobind_repro` 的 PyTorch `2.1.0+cu118` 可见 8 张 RTX 4090 D；只读预检时
  0–7 均有其他用户进程，GPU smoke 将等待人类显式给出可用 id，不猜测或抢占。
- 结果汇总 schema 升级为 1.1：终检重新验证 dependency、resolved config、
  protocol/data/commit、checkpoint/raw prediction hash 与 source singleton；输出逐 seed、
  逐 session、macro/micro、session sample std 和四组配对 bootstrap 95% CI/effect size。
- FrameAcc 原始 JSON 现在保存 source 候选相似度/Hungarian assignments，以及 Custom
  window/frame assignments；Markdown 报告由已验证 JSON 自动展开全部 20 个 session 行。
- 正式 `environment.yml` 修正项目名/激活名并声明 `yacs/pytest/ruff`；删除不完整
  YACS fallback，protocol 绑定环境 hash，run record 保存实际 runtime provenance。
- 人类已确认统一 `24/16`、Custom 循环 inner validation，并授权实验 snapshot commit；
  protocol 状态已改为 `locked`。
- Custom 四折 manifest 现同时绑定 held-out session 的 7 个 segment NPZ、timestamp
  与 raw IMU CSV；正式配置的 segment root 已从旧外部项目目录迁入统一 `/data`
  资产边界，复制前后 segment SHA256 一致，manifest 独立重建逐字节一致。
- 根 `run_pipeline.py` 增加 shebang/可执行位，在激活环境后可直接运行；新增 canonical
  frame-id 单调性与真实 checkpoint save/load/forward round-trip 契约。
- 清除通用 AlphaPose/ByteTrack/WHAM Python adapter 和 component fragments 中的
  个人路径默认值；后端 repo/weight 改为显式配置或环境变量，缺失时 fail-loud。
- 当前自动验证：`75 passed`；70 个公共模块 import、26 份 workflow config 加载、
  compileall、全量生产/工具/测试 Ruff、`git diff --check`
  和官方文档 stale-command 扫描通过；
  Custom 四折修正缓存的独立 preprocess 复用 CLI 全部通过。

### 当前状态

- E2 核心兼容层与 checkpoint/artifact 契约完成。
- E3 的三数据集 extractor backend smoke 已完成；协议锁定与训练前 manifest 冻结尚未完成。
- 正式 GPU 训练仍等待 `24/16` 窗口和 Custom 循环 inner validation 协议由人类最终锁定。

### 下一步

1. 已收到人类对统一 `24/16`、循环 inner validation 和 snapshot commit 的确认。
2. 创建并验证实验 snapshot commit，随后生成 commit-bound protocol hash 与全部 resolved configs。
3. 人类指定可用 GPU 后运行单 seed/单 fold GPU smoke，记录峰值显存与时间。
4. smoke 通过后扩展全部 required cells。
