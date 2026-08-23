# G12 Goal：人物朝向感知的 IMU–骨架配对

## 阶段目标

建立可审计的人体朝向表示，研究身体朝向变化、局部关节旋转和 IMU 角速度对人员配对的影响。G12 不把 wrist gyro 直接等同于 body yaw rate，而是区分：

```text
body/root orientation
  + torso heading
  + sensor-to-body relative rotation
  + wrist/arm local angular velocity
  -> IMU–skeleton pairing
```

## 核心任务

1. 优先盘点我们自己的 extractor raw、algorithm output 与 canonical pair input（`/data/lyxie` 下 YOLO-Pose high、AlphaPose、FMPose3D、MotionAGFormer、TCPFormer、WHAM）的 orientation 字段、坐标系、采样率和 provenance；数据集原生字段只作 reference/control。
2. 区分直接 orientation、由 3D 关节推导的 heading，以及只能作为 2D proxy 的弱信号。
3. 测量 body/torso orientation change 与 IMU gyro 的时间、频率和相位关系。
4. 在固定配对协议下，对 root yaw、torso heading、segment orientation 和 relative rotation 做增量消融。
5. 明确 orientation 缺失时的行为，不把当前 Custom 2D skeleton 当成世界朝向真值。

## 数据边界

- 当前六类 extractor canonical cache 均只保留 `skeleton/extract_skeleton (T,N,17,3)`；YOLO/AlphaPose 是 2D proxy，FMPose3D/MotionAGFormer/TCPFormer/WHAM canonical 可派生 3D heading。
- WHAM raw 产物含 `root_orient`/`pose_world` 直接 orientation 候选，但当前 adapter 未传递；不能当成模型已使用。
- TotalCapture Vicon/SMPL-X、EgoHumans 完整 SMPL `global_orient` 仅是 source-orientation reference/control。
- 当前 Custom corrected cache 只有 2D skeleton；仅作为 orientation-missing/control track。

## 成功标准

- 建立 orientation inventory、字段语义和稳定内容 hash。
- 通过 quaternion/axis-angle/6D、时间差分、wrap 和坐标系契约测试。
- 在有直接 orientation 的 source 上完成 body heading 与 gyro 的物理关联审计。
- 任何配对提升都报告 raw `correct/total`、FrameAcc、seed 方差、orientation coverage 和 provenance；没有 Custom orientation ground truth 时不宣称 Custom 正式收益。

## 主要产物

- `formulation.md`
- `survey.md`
- `ideas.md`
- `plan.md`
- `E1:skeleton_orientation_inventory/`
- 大型审计 artifact：`/data/fzliang/reid-project/g12/`
