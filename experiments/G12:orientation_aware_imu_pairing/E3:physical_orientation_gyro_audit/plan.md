# E3 Plan：Extractor 朝向变化与 IMU 角速度关系

## 正确的数据角色

- extractor skeleton：只提供 2D proxy 或 3D torso heading；
- independent IMU：提供同一人物、同一序列、同一帧的 LeftWrist gyro；
- 不要求 skeleton NPZ 自带 acceleration/gyro 字段。

## 审计步骤

1. 读取六种 S06 `algorithm_outputs`，按 `custom_XX_YYY` 对应 EgoHumans realistic-IMU `XX_YYY_aria*.npy`。
2. 使用原转换脚本的 sorted aria order 固定 person mapping，并要求 frame count 完全一致。
3. YOLO/AlphaPose 计算 π-periodic 2D shoulder-line proxy；其余方法从 H36M17 joints 计算 3D torso heading。
4. 比较 heading rate 与 gyro xyz、`abs(rate)` 与 gyro norm；同时报告 raw 和 0.25 秒平滑结果。
5. 运行 ±1 秒 lag、time shuffle、shuffled person 与 gyro-motion-stratum controls。
6. 只进行物理关联审计，不运行 pairing/re-ID ablation。

入口：

```bash
conda run -n reid_project python \
  'experiments/G12:orientation_aware_imu_pairing/E3:physical_orientation_gyro_audit/scripts/run_extractor_imu_join_audit.py'
```
