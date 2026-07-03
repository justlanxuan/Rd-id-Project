# G5 Plan: Cross-Dataset Transfer 路线图

## 总体策略

采用 **先验证架构迁移可行性，再系统比较迁移策略** 的方式。第一阶段聚焦 E1：将 G4/E11 的 dual-embedding 架构迁移到 EgoHumans → custom 场景，建立新的跨数据集基线。

## Phase 1: 源域预训练（E1-A1/A2）

### E1:egohumans_dual_embedding_pretrain

- **目标：** 在 EgoHumans 上训练 local + global dual-embedding 模型，作为源域预训练权重。
- **内容：**
  - **A1: 数据准备**
    - 确认 EgoHumans MoBInd cache 可用（G_egohumans/E6/E8 已生成）。
    - 分别构建 local（`motion_type=wjoint`, `limb_list=[RightWrist]`）和 global（`motion_type=pose2d`）的 cache。
    - 注意：global branch 需要 full pose2d，EgoHumans 上此前未系统训练过该 setting。
  - **A2: 源域训练**
    - 训练 Model-L：单 IMU RightWrist ↔ RightWrist motion，复用 G3/E2 的 Stage1/Stage2 base config，数据根目录改为 EgoHumans。
    - 训练 Model-G：单 IMU RightWrist ↔ full pose2d，复用 G4/E11 的 Stage1/Stage2 base config，数据根目录改为 EgoHumans。
    - 跑 6 seeds（0/42/123/1/2/3）或至少 3 seeds，记录 EgoHumans test 性能。
  - **A3: 源域融合评估**
    - 对每 seed 做 α sweep，记录 best α Fusion 性能。
    - 确认源域 dual-embedding 在 EgoHumans 上是否也能带来提升。

**预期产出：** `E1:egohumans_dual_embedding_pretrain/results/source_results.md`

## Phase 2: 目标域迁移（E1-A4/A5/A6）

- **目标：** 将 E1 训练好的源域 dual-embedding 权重迁移到 custom，对比 zero-shot 与多种 fine-tune 策略。
- **内容：**
  - **A4: Zero-shot 评估**
    - 直接加载 EgoHumans Model-L 和 Model-G checkpoint，在 custom test 上评估 FrameAcc。
    - 尝试 fusion（α sweep），记录 best α 性能。
  - **A5: Full fine-tune**
    - 加载 EgoHumans Stage1+Stage2 权重作为初始化，在 custom 上完整训练 Model-L 和 Model-G。
    - 与 G4/E11 from-scratch 对比。
  - **A6: Partial fine-tune / adapter**
    - 冻结 Stage1 encoder，只 fine-tune Stage2。
    - 或：在 IMU/motion encoder 后加 adapter，只训练 adapter。
    - 与 A5 full fine-tune 对比。

**预期产出：** `E1:egohumans_dual_embedding_pretrain/results/transfer_results.md`

## Phase 3: 分析与决策

- 汇总源域性能、zero-shot 性能、各 fine-tune 策略性能。
- 与 G4/E11 from-scratch 进行统计对比（mean ± std，per-clip 分解）。
- 判断：
  - 是否继续深入 I2–I8？
  - 哪种迁移策略最值得扩展（如扩展到 w100、多源数据）？

## 风险控制

| 风险 | 应对 |
|---|---|
| EgoHumans pose2d cache 不存在或格式不兼容 | 先检查 G_egohumans/E6/E8 cache，必要时复用 G3/E2 的 cache 构建脚本 |
| 源域训练耗时过长 | 优先在 w24 上训练，有效后再扩展 w100 |
| fine-tune 破坏源域表示 | A6 先尝试冻结 Stage1 / adapter，再逐步解冻 |
| custom 数据量小导致 fine-tune 过拟合 | 监控 val top1，使用 early stopping，必要时减小 learning rate |
| 与 G4/E11 的数据/脚本不一致 | 严格复用 G4/E11 的 custom 数据路径与评估脚本 |

## 时间/资源估算

- A1 数据准备：~30 分钟（若 cache 已存在则更快）。
- A2 源域训练：每 seed 每 branch ~20–40 分钟，6 seeds × 2 branches ≈ 4–8 GPU·小时，可并行压缩至 1–2 小时。
- A3 源域融合评估：~30 分钟。
- A4 zero-shot：~10 分钟。
- A5/A6 fine-tune：与 A2 相当。
- 总计：约 1–2 个工作日（含调试）。
