# E4.1 Initial Audit：AlphaPose 朝向能力与当前缺口

## AlphaPose 能提供什么

AlphaPose 输出的是 2D image-plane keypoints 和 confidence。左右肩线、左右髋线、肩髋相对几何可以生成朝向 proxy；但没有 root quaternion、3D torso frame 或 camera/world yaw。因此它是弱 orientation cue，不是直接 orientation measurement。

## Custom23 直接审计

在 corrected Custom23 sequence `20260211_171423` 上，使用现有 AlphaPose skeleton 和 measured IMU 6D sequence，对 shoulder-line proxy rate 与 LeftWrist gyro 做全序列审计：

| person | smoothed max `|corr(rate, gyro_axis)|` | `corr(|rate|, ||gyro||)` |
|---:|---:|---:|
| 0 | `0.0615` | `0.0547` |
| 1 | `0.0948` | `0.0092` |

这比 E3 在 EgoHumans realistic-IMU 上 AlphaPose matched median `0.198` 弱很多，说明把 AlphaPose 2D shoulder angle 直接作为 Custom23 的方向真值是不够的。可能原因包括 camera-view dependence、肩部检测噪声、wrist gyro 的局部旋转成分、source-target extractor/coordinate mismatch 和全序列中转向只占少量时间。

## 当前结论

1. AlphaPose 不是 orientation-free，但可观测方向主要是弱 2D proxy。
2. 当前 turning-gate 没有明显提升，不能简单归因于“模型没关注方向”；更可能是输入方向信号本身弱且 source-target 不同源。
3. 提升优先级应是：同源 extractor contract → 3D lifting heading → turn-stratified target → cross-attention/辅助损失，而不是继续堆叠同一个 2D angle 的融合层。

详细实验矩阵见本目录 `plan.md`；机器审计摘要位于 `/data/fzliang/reid-project/g12/e4_orientation_model/e4_1_alphapose_custom23_audit.json`。

## E4.1 同源模型 screen（2026-08-21）

为排除先前 `gt` 训练骨架与 Custom AlphaPose 测试骨架不一致的问题，新建了
`motionbert_alphapose_cache_v2`：TC 训练和 Custom 测试均使用 MotionBERT 的
AlphaPose-derived `keypoints_h36m.npy`，并保留 `[T,17,3]` 的坐标/置信度语义。
Custom 窗口按 `source_person` 切片，避免两名候选误读同一人的序列。严格同源
训练使用 TC-only；TC+EgoHumans 混合结果只作为 balanced control，不进入主结论。

3D heading 使用 MotionBERT 三维坐标（up-axis=`y`）派生 torso heading；2D proxy
只使用 image-plane shoulder line。所有数值为 3 seeds、3 epochs×50 steps 的
screen；checkpoint 按 Custom23 high-turn 选择，因此是探索性结果，不能视为无偏
测试集估计。

Cache manifest SHA256=`ccf9425387ca3ac73b646d0b62457f7795817d48ab60493afc1d81a76596ac99`；
初始 AlphaPose-IMU audit SHA256=`c63e7da87aa48e309d784aeb044af0ce0e7407e581c0bb955c54e4f6bfa957de`。

| 同源表示/模型 | Custom23 high-turn | Custom23 low-turn | Custom23 全部 | 57/22/24 全部（依次） |
|---|---:|---:|---:|---:|
| 2D proxy baseline | 0.487±0.038 | 0.478±0.022 | 0.480±0.018 | 0.502 / 0.522 / 0.497 |
| 2D proxy turning-gate | 0.513±0.047 | 0.496±0.011 | 0.500±0.004 | 0.505 / 0.515 / 0.507 |
| 3D heading baseline | 0.482±0.039 | 0.475±0.031 | 0.477±0.013 | — |
| 3D heading turning-gate | 0.524±0.030 | 0.484±0.018 | 0.495±0.015 | — |
| 3D heading gyro cross | 0.518±0.025 | 0.551±0.026 | 0.542±0.012 | 0.500 / 0.495 / 0.491 |
| 3D heading gyro cross + learnable gyro-activity auxiliary | **0.583±0.059** | 0.461±0.020 | 0.495±0.004 | 0.501 / 0.498 / 0.488 |

这里的 auxiliary 已修正为可学习项：朝向分支的窗口 embedding 预测该窗口的
陀螺模长活动；此前直接比较输入朝向和输入 IMU 的实现没有梯度，已废弃。结果
支持“3D heading + gyro cross + activity supervision 能把收益集中到转向片段”，
但同时显示低转向退化，尚未达到晋级标准。2D proxy 的 high-turn 提升很小且方差
大，说明 AlphaPose 的图像平面方向不足以稳定解释 Custom23 的 IMU 转向。

负控 session 的准确率仍围绕 0.5，未观察到普遍收益；由于训练选择使用 23 的
high-turn，负控数字只用于检查是否出现系统性副作用。下一步需要冻结 epoch/验证
集、增加 seeds，并加入 hard-negative 与 turn-onset loss 后再做确认实验。

### Invalidated conditional-cross run

首次 conditional-cross validation run 被立即中止并作废：实现曾用 skeleton-derived
turning activity 同时控制 skeleton 和 IMU residual，导致 IMU embedding 间接读取
配对骨架信息，违反双塔检索隔离。异常高 validation 数字不得引用。修正版 skeleton
gate 只读 orientation，IMU gate 只读 gyro embedding，并用 modality-isolation test
验证一侧输入变化不会改变另一侧 embedding。

### Superseded candidate-level strata

冻结确认后审计发现旧 `evaluate()` 虽然用 group-mean activity 计算阈值，却用单个
candidate activity 决定 high/low，导致同一候选组的两行可能落入不同 stratum。
旧表的 full-session 数字仍有效，但旧 high/low 数字标记为 superseded。修正版按
candidate group 统一分层；physical turning-MoE 以及后续重算均使用该口径。

## Authoritative group-level confirmation

最终协议使用无重叠 Custom23 validation/test、validation-only turning threshold
`19/48`，并用修正后的 group-level stratum 重新选择 5 个 baseline checkpoints。
学习式 concat/gate/cross、window auxiliary、turn-weighted InfoNCE、25/50/100% hard
negatives 和 8-bin turn-onset loss 均未形成稳定晋级结果；100% hard negatives 还会
接近 InfoNCE chance/collapse。

有效设计是显式 physical turning mixture-of-experts：高转向组用
`max_{lag=-2..2} corr(|3D heading rate|, gyro magnitude)` 做候选打分，低转向组完全
回退 frozen baseline。3D heading 来自 MotionBERT lifting，不是 AlphaPose 直接提供
的 root/world orientation。

| corrected group-level 指标 | baseline | physical turning-MoE | 配对差值（95% CI） |
|---|---:|---:|---:|
| Custom23 test high-turn（56 queries） | 0.471±0.021 | **31/56=0.554** | **+0.082 [ +0.052, +0.112 ]** |
| Custom23 test low-turn（44 queries） | 0.509±0.023 | 0.509±0.023 | 0.000 |
| Custom23 test 全部（100 queries） | 0.488±0.015 | **0.534±0.010** | **+0.046 [ +0.029, +0.063 ]** |
| Custom57 全部 | 0.503±0.010 | 0.503±0.010 | 0.000 |
| Custom22 全部 | 0.486±0.010 | 0.486±0.010 | 0.000 |
| Custom24 全部 | 0.502±0.011 | 0.497±0.011 | -0.005 |

Custom24 的下降来自两个孤立的 3D-heading activity spike。查看负控后提出的
persistence safety router 要求 ±24 source frames 内存在另一个 high group；它在
Custom23 test 保持/略增收益（high `0.557±0.018`，配对差 `+0.086`，95% CI
`[+0.057,+0.115]`），同时在 57/22/24 完全回退 baseline。由于 persistence 规则是
查看 control 后提出的，只能标记为 post-hoc safety candidate，仍需新的独立转向
session 才能成为 release-level promotion。

权威机器摘要：`/data/fzliang/reid-project/g12/e4_1_confirmation_summary.json`
（SHA256=`6a00101726a199afc759df1b399a6953caa042b8cf8d2f6a28dbdf70b64b5c01`）；
MoE 原始结果：`/data/fzliang/reid-project/g12/e4_1_physical_turning_moe.json`。
