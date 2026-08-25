# G13 E4：H4W++ 3-D skeleton / 朝向特征 profiling

## 目的

在 E3 确定全帧推理为稳健密度后，固定 H4W++ full-frame skeleton，系统测试原有
2-D skeleton 与新的 H36M-17 3-D 信息：深度、速度、3-D bone vectors、显式 torso
heading、heading rate、heading-invariant 坐标和骨段几何。

## 固定变量

- 四折 LOSO：每折三个 session 训练，剩余一个 session 测试。
- H4W++ full-frame cache；不改变推理密度。
- Window length/stride：`24/16`。
- Hybrid IMU/skeleton matcher，20 epochs，batch size 64，`best_metric: train_top1`。
- Canonical IMU person mapping，`multi_person: true`，两个测试候选。
- 训练 fold 计算 IMU 与 skeleton feature statistics；不使用 val/test session 统计。
- FrameAcc 为主要选择指标；不使用测试结果选择 checkpoint。

## 扫描变量

每个 feature 做 seeds `0/42/123` 的四折 LOSO，共 `14 × 3 × 4 = 168` runs：

| feature | 定义 |
|---|---|
| `hybrid` | 现有 2-D shoulder/local-arm baseline；只取 skeleton 的 x/y |
| `h36m2d` | H36M-17 root-relative position + velocity，z 置零；与 `h36m3d` 同一 102-D tower |
| `h36m3d` | H36M-17 root-relative、肩宽归一化的 x/y/z position + velocity |
| `h36m3d_no_velocity` | 同一 3-D position，velocity 通道置零 |
| `h36m3d_bone` | 3-D parent-to-joint bone vectors + velocity |
| `h36m3d_heading` | 3-D position/velocity 加 torso heading 的 sin/cos、rate、validity |
| `h36m3d_heading_rate` | 仅加入 heading rate 与 validity，不加入绝对 heading |
| `h36m3d_rotinv` | 按每帧 torso heading 对 x/z 坐标旋转到 canonical heading |
| `h36m3d_zonly` | 仅保留 root-relative depth z，x/y 置零；保留 velocity |
| `h36m3d_geom` | 3-D position/velocity 加 17 个骨段长度通道 |
| `h36m3d_left_wrist` | 左前臂 3-D 单位方向 + frame-to-frame rotation proxy |
| `h36m3d_right_wrist` | 右前臂 3-D 单位方向 + rotation proxy，作为左右侧对照 |
| `h36m3d_both_wrist` | 左右前臂方向与 rotation proxy |
| `h36m3d_left_wrist_rot` | 左前臂方向投影到 torso frame（lateral/up/forward）+ rotation proxy |

H4W++ 输出已经在 extractor adapter 中映射为 H36M-17：`pelvis, hips, knees, ankles,
spine, thorax, neck, head, shoulders, elbows, wrists`。heading 使用
`forward = (right_shoulder-left_shoulder) × (thorax-pelvis)`，Y 为 up，水平面为 X/Z；
该坐标约定和 heading 方向会写入结果 manifest。

当前 cache 不含 MANO/手指关节，因此 wrist 消融严格标记为 forearm/wrist direction
proxy，不能解释为完整的 wrist-local rotation。只有在该 proxy 显示稳定收益后，才追加
保留 SMPL-X/MANO hand joints 的 extractor 版本。

## 统计顺序

1. 每个 `feature × seed` 完成四个 held-out session；
2. 每个 seed 对四折 FrameAcc 做等权宏平均；
3. 每个 feature 对三个 seed 的宏平均再做等权平均，并报告 population std；
4. 以三 seed 四折宏平均最高者作为 profiling 最优；同时保留 weighted FrameAcc；
5. 只有 `h36m2d` vs `h36m3d` 是严格同 tower 的深度增量对照，其他 feature 结果解释为
   受控表征候选，不把不同输入宽度的绝对分数当成纯信息因果估计。

## Artifact 边界

- 输入 cache：`/data/fzliang/reid-project/custom/preprocessed/h4wpp_fullframe_w24/`
- 训练/测试：`/data/fzliang/reid-project/custom/artifacts/h4wpp_3d_feature_sweep/`
- configs、逐 run checkpoint/results、summary 和 SHA-256 manifest 均保存于 artifact root。
