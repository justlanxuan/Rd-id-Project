# E4.1 Test Contract

- 所有 orientation features 必须标注 extractor、坐标系、关节 contract 和 confidence/missingness provenance。
- AlphaPose 2D proxy 不得命名为 world yaw 或 root orientation。
- source train 与 Custom23 的 extractor 必须一致；若不一致，结果只能标为 mismatch control。
- Custom23 high-turn/low-turn 分层阈值须在训练前冻结，不能用测试正确率反推。
- 主要结果必须同时报告 high-turn Custom23、全 Custom23、low-turn Custom23 和 57/22/24 negative controls。
- 3D lifting、WHAM raw orientation 和 2D proxy 必须分开表格，禁止合并成“orientation”单一结果。
- 任何辅助 gyro loss 只能约束可审计的 rate/magnitude 关系；未知坐标变换下禁止强制 axis-to-axis regression。
- 辅助损失必须作用于模型输出（不能只对输入 orientation/IMU 计算常数项）；当前实现
  为 orientation embedding → gyro-activity prediction，并在日志中保留训练梯度。
- 双塔隔离：skeleton/orientation 输入变化不得改变 IMU embedding/gyro onset 输出，
  IMU 输入变化不得改变 skeleton embedding/orientation onset 输出；pairwise physical
  correlation 只能发生在最终候选 score 层。
- high/low 必须以 candidate group 统一分层；activity 使用 2×24 个二值位的整数计数，
  threshold=`19/48`，禁止 float32 边界决定分层。
