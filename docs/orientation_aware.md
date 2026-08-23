# 人物朝向变化

本实验把“人物朝向”定义为**我们自己的骨架提取器输出推导出的时序特征**，而不是数据集附带的朝向标签。每个窗口仍然使用原始 skeleton/IMU 两路输入，额外计算 5 维 orientation stream：

`periodic_sin, periodic_cos, rate_scaled, orientation_valid, turning_activity`

其中 `rate_scaled` 来自骨架方向的时间变化，`turning_activity` 是其绝对值；IMU 角速度仍来自 IMU sidecar 的 gyro xyz。默认窗口为 24 帧、0.8 秒，在线使用时可按相同窗口长度滑动，模型本身没有跨窗口未来信息。

## 主框架开关

```yaml
TRAIN:
  MODEL:
    TYPE: orientation_aware
  ORIENTATION:
    ENABLED: true
    MODE: 3d_heading       # proxy | 3d_heading | none
    PROFILE: full          # full | rate
    FUSION: cross           # gate | concat | gyro_focus | cross | conditional_cross | residual
    TRAIN_SPECS:
      - dataset=custom23;csv=/path/windows.csv;root=/path/windows;fps_hz=30;gyro_sidecar_root=/path/gyro
    VAL_SPECS: []
    TEST_SPECS: []
  # 可选：turning 加权 InfoNCE、gyro/turning 对齐、onset 辅助损失
  # ORIENTATION.AUX_TURNING_WEIGHT: 0.0
  # ORIENTATION.TURNING_LOSS_WEIGHT: 0.0
  # ORIENTATION.TURN_ONSET_WEIGHT: 0.0

TEST:
  METRICS:
    TURNING_MOE:
      ENABLED: true
      THRESHOLD: 0.3958333333  # 19/48，保持 E4.1 预注册协议
      MAX_LAG: 2
```

`TRAIN.MODEL.TYPE=hybrid`（默认）完全保持原有数据和 checkpoint 行为。G12 旧脚本仍可复现实验，但新的代码应通过 `src.models.registry`、`src.datasets` 和 `src.metrics.turning` 使用公共接口。`src/g12/*`、`tools/g12/*` 中保留的名称是兼容层，不再被官方 train/evaluate 入口反向依赖。

## 已验证的实验结论

- 只有 Custom session 23 包含可观测转向，其他 session 理论收益接近零；因此结果必须按 high/low turning 分层，不能只看总体平均。
- E4.1 physical turning-MoE 仍是当前 Custom23 最强的冻结基线增强：full 约 `0.534`，high 约 `0.554`；low 组保持 baseline。它用 `max_lag_pearson(abs(orientation_rate), gyro_magnitude)`，只在 turning count ≥ 19 的组路由到 physical expert。
- E5 fully-aligned O3D `turning_cross` 是新的可训练候选：Custom23 full 相对 baseline 约 `+2.0pp`，high 约 `+0.4pp`，low 约 `+4.1pp`，但 EgoHumans 验证没有稳定收益。因此它是候选模型，不替代 E4.1。
- O2D proxy、TC+Ego balanced 和未分层的全局平均没有显示稳定增益；继续实验时应优先保持 source-aligned skeleton、固定窗口、固定 seed，并单独报告 session 23 high/low。

## 兼容和审计

- 原有 `video` matcher 输出键保留；orientation 模型同时返回历史 `skeleton` 键。
- raw spec 语法由 `src.data.specs` 统一解析；G10/G12 工具的私有解析器和 sampler 已改为兼容别名。
- physical turning 只负责推理后处理，训练损失和评估路由分离，避免把 IMU 角速度伪装成骨架输入。
