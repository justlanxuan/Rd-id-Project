# E2/E4/E5 Plan：IMU、动作复杂度、跨模态与跟踪

## Scope

只使用 E1 `included` 子集：TotalCapture GT、EgoHumans canonical、Custom canonical、S06 AlphaPose、FMPose3D、MotionAGFormer、TCPFormer、WHAM。YOLO-Pose high 保留为 conditional，不进入第一版归因排名。

## Measurements

- B1 对每个文件/人员计算 bone-scale normalized motion energy、速度峰值、加速度、jerk、active-joint ratio、simultaneous-motion ratio、谱熵和周期性；同时计算 IMU acceleration energy/jerk 及布局统计。
- B1 在 `lag=-8..8` frame 上计算 wrist-speed 与 IMU acceleration magnitude 的 screening correlation；正 lag 定义为 `skeleton[t+lag]` 对 `IMU[t]`。
- B1 明确记录 7D `acc3+quat(wxyz)`、legacy 48D rotation-matrix+acceleration 以及 Custom raw CSV 的 10 Hz→约 30 fps 重采样证据。
- C1 将 legacy48 通过已存在的 `legacy_imu48_sensor_to_7d(L_LowArm)` 转为同一 7D contract，并单独记录 quaternion norm invalid tail；转换不覆盖原始数据。
- B2 对 S06 visibility coverage、candidate group size、tracklet run length、fragmentation 和 baseline visibility delta 分层；独立 ID switch 仅在存在独立 track IDs 时才声称可测。

## Interpretation guardrails

1. 2D xy、2D xy+visibility 和 3D xyz 分轨，不能把最后一维长度 3 直接当作 3D。
2. motion magnitude 只在单记录内按骨长归一化；不同原始坐标空间不做直接 pooled 数值排名。
3. lag correlation 是筛选证据，不是因果结论；正式模型干预前必须保留 session/frame provenance。
