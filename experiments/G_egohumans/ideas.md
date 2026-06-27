# 💡 G_egohumans 想法孵化（整理版）

基于 E1–E9 的发现，孵化下一步可探索的方向。每个想法包含动机、预期收益与主要风险。

## 已验证的关键前提

- MoBInd synthetic IMU 在 EgoHumans 上判别性很强。
- MoBInd 性能几乎完全依赖 100 帧长窗口；短窗口下单 IMU 会崩溃。
- 我们的 pipeline 在短窗口 + 单 IMU 下仍强劲，但对 IMU 数量更敏感。
- EgoHumans → custom 的域差距大，简单 frozen adapter 无法解决。

---

## I1. 在 custom 上验证多 IMU 的效果

**动机：** E5/E6 显示我们的 pipeline 对 IMU 数量敏感。E9 为了和 E8 源模型一致只用了单 IMU，但 custom 数据本身有多个 IMU 传感器。

**做法：**
- 在 custom 4-fold 上训练 4-IMU 版本的 A2/A4（保留 E8 100 帧源模型用于 A4）。
- 与 E9 单 IMU 结果对比。

**预期收益：**
- 若 4-IMU 显著提升，则确认我们的 pipeline 在真实数据上也依赖传感器数量，而非仅 EgoHumans 特有。
- 为 custom 实际部署选择输入配置提供依据。

**风险：**
- Custom 数据 IMU 位置与 EgoHumans synthetic 不完全一致，可能需要 sensor 映射。

---

## I2. 更强的域自适应方法（替代 frozen affine adapter）

**动机：** E9 A3 的 96 参数 affine adapter 完全无效。需要可学习的域自适应模块。

**候选方案：**
1. **DANN / gradient reversal：** 在 IMU 编码器后加 domain classifier，让特征对 EgoHumans vs custom 不可区分。
2. **CORAL / MMD 对齐：** 最小化源域与目标域 IMU 特征分布的二阶矩差异。
3. **BN adaptation：** 仅更新 target domain 的 BN 统计量或 affine 参数。
4. **IMU 统计对齐：** 不只是 z-score，而是学习一个 per-channel affine + shift（比当前 96 参数更大，或在 embedding 空间做）。

**预期收益：**
- 显著提升 zero-shot / few-shot transfer，可能超过 from-scratch。

**风险：**
- 需要实现新的训练逻辑；超参搜索成本高；小目标数据集容易过拟合。

---

## I3. EgoHumans + custom 联合预训练

**动机：** E9 显示 EgoHumans 预训练直接迁移有限，但提供了良好的初始化。结合源域大数据与目标域小数据可能更好。

**做法：**
- 阶段 A：在 EgoHumans 上预训练（已完成，E8 checkpoint）。
- 阶段 B：在 EgoHumans + custom 混合数据上继续训练若干 epoch（域标签保留，用于域自适应或仅作为数据增广）。
- 阶段 C：在每个 custom fold 上 fine-tune。

**变体：**
- 是否使用域标签？
- 是否对两个域分别计算 IMU stats？

**预期收益：**
- 利用 EgoHumans 的动作多样性增强 custom 模型的泛化；可能比纯 custom from-scratch 更稳定。

**风险：**
- 混合训练可能让模型偏向数据量大的 EgoHumans；需要 careful sampling 或 domain weighting。

---

## I4. 分析 custom 各 session 的难度差异

**动机：** E9 fold2 在单 seed 下被模型完全做对，而 fold1/fold3 明显更难。理解这种差异有助于数据收集与模型设计。

**分析维度：**
- 每个 session 的人数、时长、动作类别多样性。
- IMU 信号方差、传感器缺失/噪声情况。
- 视频骨架质量（检测置信度、遮挡比例）。
- 类别不平衡程度（某些身份窗口数远多于其他）。

**预期收益：**
- 识别导致难/易 session 的数据特征。
- 为 future data collection 提供指导（如增加动作多样性、减少遮挡）。

**风险：**
- 需要额外可视化/统计脚本，属于分析性工作。

---

## I5. 在 custom 上测试窗口长度敏感性

**动机：** E8 发现我们的 pipeline 在 EgoHumans 上对窗口长度不敏感。需要确认这在 custom 真实数据上是否同样成立。

**做法：**
- 重新 slice custom 数据为 12/24/48/100 帧窗口（保持 cross-session 4-fold）。
- 训练 A2 from-scratch 与 A4 finetune，比较 FrameAcc 与 grouped-test。

**预期收益：**
- 如果短窗口仍强劲，可在 custom 部署中使用更小的延迟窗口。
- 如果长窗口显著提升，说明真实数据需要更多时序上下文。

**风险：**
- 重新 slice 并训练 4 folds × 多个窗口 × 多 seed，计算量较大。

---

## I6. Synthetic-to-real IMU 适配

**动机：** EgoHumans 使用 MoBInd synthetic IMU，custom 使用真实 IMU。两者在幅值、噪声、漂移上差异可能是域差距的主要来源。

**做法：**
- 可视化并量化 synthetic vs real IMU 的 per-channel 分布差异。
- 尝试在输入层或特征层做分布对齐（如 adversarial domain adaptation 仅针对 IMU encoder）。
- 或尝试在 synthetic 数据上加真实风格的噪声/漂移增强。

**预期收益：**
- 直接针对 E9 中观察到的域差距来源，可能比端到端 finetune 更高效。

**风险：**
- 需要获取或估计真实 IMU 噪声模型；增强策略可能过拟合。

---

## I7. 跨三数据集预训练（TotalCapture + EgoHumans + custom）

**动机：** 当前 pipeline 已在 TotalCapture 上验证。若将三个数据集统一预训练，可能获得更通用的身份表示。

**做法：**
- 统一数据格式与 IMU 传感器映射。
- 多数据集联合训练，可能加上 domain-specific BN 或 domain classifier。
- 在 custom held-out session 上测试迁移。

**预期收益：**
- 最大化利用所有数据，提升对 custom 的泛化。

**风险：**
- 数据异构性大（实验室 vs 真实场景 vs ego-centric）；实现复杂度高。

---

## 优先级建议

| 优先级 | 想法 | 理由 |
|---|---|---|
| P0 | I2 域自适应 / I3 联合预训练 | 直接针对 E9 未解决的核心问题 |
| P1 | I1 custom 多 IMU | 工程成本低，能快速验证输入敏感性 |
| P2 | I4 session 难度分析 | 解释性强，指导后续数据/模型决策 |
| P3 | I5 custom 窗口长度 | 验证 E8 结论是否泛化到真实数据 |
| P4 | I6 synthetic-to-real / I7 三数据集 | 更长期，收益高但实现复杂 |
