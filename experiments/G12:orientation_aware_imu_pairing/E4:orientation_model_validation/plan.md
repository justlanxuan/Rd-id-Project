# E4 Plan：通过模型显式关注转向

## 问题

E3 证明 extractor-derived torso orientation 的变化与同一人物的 IMU gyro 有物理相关性；E4 检验这种信息能否转化为跨模态匹配收益。骨架只提供朝向，gyro 只从独立 IMU sidecar 读取，不把 gyro 写入 skeleton。

## 模型设计

- baseline：沿用 G11 skeleton encoder + IMU encoder + InfoNCE，不消费 orientation stream。
- turning-gate：从未做 bbox normalization 的 extractor skeleton 派生 2D shoulder-line proxy：`sin(angle), cos(angle), clipped rate, valid, turning_activity`；用独立 TemporalEncoder 编码，并用 learned sigmoid gate 调节其注入 skeleton embedding。
- turning-concat/gyro-focus/residual：作为实现级单 seed sanity/negative controls；主 screen 只冻结 baseline 与 turning-gate，避免把多个自由度混入首轮结论。

## 固定协议

- 训练：TotalCapture + EgoHumans realistic-IMU，按现有 DomainBalancedGroupBatchSampler；相同 source manifests、normalization、candidate groups 和 seed `0,1,2`。
- 评估：同一 Custom complete-session candidate groups（23、57、22、24），不改变 identity mapping、窗口 stride 或评估代码。按动作语义，23 是转向收益目标；57/22/24 是非转向负控，只检查不应产生系统性退化。
- 窗口：正式比较 0.8 s（target_len=24）和探索性 2.0 s（target_len=60）；其余模型超参相同，3 epochs × 50 steps。
- 选择：每个 seed 仅按 Custom23 development split 的最高 FrameAcc 选 epoch，再报告冻结的 57/22/24；不使用测试 session 反向调参。

## 通过门槛

至少三 seed 的 Custom23 均值相对 baseline 上升，且不能在非转向负控上产生系统性退化；同时记录 raw correct/total、margin、gate 分布、参数/配置和 artifact hash。单 seed 峰值只作为探索信号，不作为 promotion。

## Source-aligned follow-up（根据反馈新增）

当前 E4 的训练 manifests 是 `skeleton_source=gt`，而 Custom23 使用 AlphaPose extractor skeleton，存在 source→target skeleton provenance mismatch。下一轮不直接复用当前 checkpoint：

1. 固定同一个 extractor；默认用 AlphaPose（Custom23 已有完整 AlphaPose artifact，且 S06 也有 AlphaPose source artifacts）。
2. source train 和 Custom23 eval 都读取该 extractor 的同一关节顺序、坐标归一化和 visibility contract。
3. 从同一 extractor skeleton 派生 torso orientation，再将 orientation stream 与 IMU gyro sidecar 配对。
4. 只把 Custom23 作为 turning target；57/22/24 只做 non-turning negative controls。

若选择 YOLO-Pose high、FMPose3D 或 WHAM 替代 AlphaPose，则必须分别生成完整 source/Custom23 同源 artifact，并独立记录结果，不能与 AlphaPose 结果合并。
