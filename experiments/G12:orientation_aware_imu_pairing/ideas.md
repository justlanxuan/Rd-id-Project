# G12 Ideas：朝向感知的 IMU 配对

## 人类提出的核心想法

现有模型似乎没有考虑人物朝向；人物朝向变化可能与 IMU 角速度相关，因此应把朝向作为独立因素纳入 IMU 配对研究。

## 已结构化方向

1. **Body heading track**：root quaternion/axis-angle、yaw sin/cos、yaw rate。
2. **Torso geometry track**：由 pelvis、shoulder 或 hip 平面推导 forward direction 和 turn rate。
3. **Local segment track**：upper-arm/forearm/wrist orientation 与 IMU gyro 分开建模。
4. **Relative rotation track**：显式计算 sensor orientation 与对应 skeleton segment orientation 的相对旋转。
5. **Physical audit first**：先做 lag、coherence、互信息和 hard-negative 对照，再做 learned pairing。
6. **Orientation ablation**：baseline、root yaw、torso heading、6D orientation、relative rotation 逐项加入。
7. **Missing-orientation track**：Custom 当前没有可靠世界朝向，必须显式报告缺失，而不是用 2D proxy 伪造真值。
8. **Sensor-placement split**：TotalCapture `L_LowArm`、EgoHumans `LeftWrist`、Custom left wrist 的 placement/provenance 分开记录。

## 关键混淆与控制

- wrist gyro 可能反映手臂摆动而非身体转向；加入 arm-only movement 和 torso-vs-wrist 对照。
- global yaw 可能受坐标系定义、相机姿态和初始 heading 影响；同时报告 absolute、window-relative 和 derivative views。
- quaternion sign flip 会制造假角速度；必须先做连续化和 wrap 测试。
- 2D shoulder/hip axis 的方向存在前后翻转和视角歧义；只作为 proxy/negative control。
- EgoHumans SMPL `global_orient` 是估计产物，不能与 TotalCapture Vicon GT 静默等同。

## 暂不采用

- 直接把 gyro z 当作 body yaw rate；
- 用 Custom 2D skeleton 生成未经验证的世界 yaw 标签；
- 在 G10/G11 checkpoint 上追加 orientation 输入并宣称协议兼容；
- 在完成 source 物理审计前运行大型 orientation-aware 训练矩阵。
