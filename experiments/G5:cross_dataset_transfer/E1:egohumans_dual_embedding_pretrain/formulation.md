# E1 Formulation: EgoHumans Pre-Trained Dual-Embedding Transfer to Custom

## 1. 目标

验证 G4/E11 的 local + global dual-embedding 架构在跨数据集场景下的可行性：
- 在 **EgoHumans** 上预训练 Model-L（local）和 Model-G（global）。
- 迁移到 **custom** 上，分别评估：
  1. **Zero-shot**：直接加载源域 checkpoint 评估。
  2. **Fine-tune**：加载源域权重，在 custom 上继续训练。
- 与 G4/E11 custom from-scratch 结果对比。

## 2. 假设

- **H1:** EgoHumans 上学习的 dual-embedding 表示具有足够的通用性，zero-shot 性能优于历史 E9 zero-shot（0.339）。
- **H2:** 源域预训练 + custom fine-tune 能够稳定超过 custom from-scratch（0.752 ± 0.095）。
- **H3:** Local branch 与 Global branch 的迁移能力不同，Global branch 可能因 full pose 更依赖域，而 Local branch 更稳定。

## 3. 关键指标

- EgoHumans test FrameAcc（源域性能）。
- Custom zero-shot FrameAcc。
- Custom fine-tune FrameAcc（mean ± std, 6 seeds）。
- 与 G4/E11 from-scratch 的 Δ。

## 4. 与 G4/E11 的关系

- 复用 G4/E11 的 dual-embedding 训练与融合脚本。
- 仅将训练数据从 custom 切换为 EgoHumans，目标域评估仍在 custom。
