# G13:H4WPP

## Goal

评估 Hand4Whole++ 3-D skeleton 是否能够提升 Custom session matching 性能，并分离
两个可能的性能来源：骨架质量与推理采样密度。

G13 固定使用严格的四折 LOSO：每折三个 session 训练，剩余一个 session 测试。
E1 是已完成的稀疏推理基线；E2 是全帧推理对照。两个实验保持训练模型、窗口、
IMU 配对、session 划分和评估指标一致，只改变 H4W++ 推理帧间隔。

## Experiments

- `E1:sparse_loso3train`：H4W++ 每 16 个视频帧推理一次，中间帧前向填充；FrameAcc
  宏平均 `80.89%`。
- `E2:fullframe_loso3train`：H4W++ 每个视频帧推理一次，不使用稀疏推理填充；FrameAcc
  宏平均 `73.79%`，低于 E1 `7.10` 个百分点。
- `E3:inference_density_sweep`：扫描 inference stride
  `1,2,4,8,12,16,24,32,48,64`，每个密度做 seeds `0/42/123` 的四折 LOSO，选择
  3-seed 平均最高的推理密度。已完成 120 runs；stride 1 最高，为
  `73.80% ± 0.41%`。

## Decision rule

E2 相对 E1 的变化用于判断稀疏推理是否是主要收益来源：

- E2 明显接近或超过 E1：性能主要来自 3-D skeleton 表征；
- E2 明显低于 E1：稀疏推理/前向填充可能是重要贡献，需进一步做密度消融。
