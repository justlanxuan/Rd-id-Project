# G13 E3：H4W++ 推理密度梯度 / 3-seed 四折 LOSO

## 目的

在 G13 E1/E2 证明推理密度会影响结果后，对 H4W++ inference frame stride 做系统
梯度测试，确定当前 Custom 协议下分数最高的推理密度。

## 固定变量

- 四折 LOSO：每折三个 session 训练、剩余一个 session 测试。
- Window length/stride：`24/16`。
- Hybrid model，20 epochs，batch size 64，`best_metric: train_top1`。
- Canonical IMU person mapping，`multi_person: true`，测试候选数为 2。
- FrameAcc 与 Group Test 配置保持不变。
- 不使用验证或测试 session 选择 checkpoint。

## 扫描变量

- H4W++ inference frame stride：`1,2,4,8,12,16,24,32,48,64`。
- 对应约 30 FPS 视频的推理频率：`30,15,7.5,3.75,2.5,1.875,1.25,0.9375,0.625,0.46875 Hz`。
- Seeds：`0,42,123`。

## 受控密度构建

E3 使用 G13 E2 已完成的全帧 H4W++ JSON，根据 `frame_id % inference_stride == 0`
确定性保留推理帧，再使用与 E1 相同的前向填充生成逐视频帧时间轴。这样所有密度
共享完全相同的模型输出，唯一差异是保留的推理帧密度；不重复调用 H4W++，避免 GPU
非确定性成为额外变量。

## 统计顺序

1. 每个 `stride × seed` 独立完成四折 LOSO；
2. 每个 seed 对四个 held-out session 的 FrameAcc 做等权宏平均；
3. 每个 stride 对 seeds `0/42/123` 的四折宏平均再做等权平均，并报告 seed 标准差；
4. 同时报告每个 stride 的全部窗口加权 FrameAcc，但最佳密度按 3-seed 的四折宏平均选择；
5. 若最高均值差异小于 seed 波动，不宣称稳定优胜，只记录候选平台区间。

## Artifact 边界

- Prepared cache：`/data/fzliang/reid-project/custom/preprocessed/h4wpp_density_w24/stride_*/loso_*`
- Train/evaluate：`/data/fzliang/reid-project/custom/artifacts/h4wpp_density_sweep/`
- Materialized configs、逐 run 结果和汇总：同一 artifact 根目录下保存。
