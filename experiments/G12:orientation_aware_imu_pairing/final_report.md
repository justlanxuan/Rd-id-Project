# G12 Final Report：人物朝向信息与 IMU 配对

## 结论

AlphaPose 不是完全没有朝向信息，但它只有 image-plane 2D keypoints；肩线/髋线只能
形成弱、视角相关的 π-periodic proxy。Custom23 上 2D shoulder-rate 与 measured
wrist gyro 的全序列相关很低（最大 axis `|r|≈0.06–0.09`），同源 2D turning-gate
也没有稳定提升。MotionBERT lifting 后派生的 3D torso heading 更适合表达人物转向，
但直接把它拼入 learned embedding、learned gate/cross、window auxiliary、turn-weighted
InfoNCE、hard negatives 或 turn-onset loss仍不稳定。

最终有效设计不是更大的融合网络，而是 turn-conditioned physical mixture-of-experts：

1. 候选组 turning count `<19/48` 时，完全使用 frozen skeleton/IMU baseline；
2. turning count `≥19/48` 时，用
   `max_lag±2 corr(|MotionBERT 3D heading rate|, gyro magnitude)` 打分；
3. skeleton 与 IMU tower 保持隔离，跨模态 correlation 只发生在最终 pair score。

在无帧重叠、validation-only 阈值、group-level 分层的 Custom23 frozen test 上，5 个
重新选择的 baseline checkpoints 为 high-turn `0.471±0.021`；physical turning-MoE
为 `31/56=0.554`，配对提升 `+8.2pp`，95% CI `[+5.2,+11.2]`。low-turn 完全不变；
全 test 从 `0.488±0.015` 提升到 `0.534±0.010`。

57 和 22 没有窗口触发 physical expert，因此结果与 baseline 完全相同。24 有两个
孤立 lifting spike 被误触发，造成全体约 `-0.5pp`。查看负控后提出的 persistence
router（±24 frames 内必须存在另一个 high group）消除了这些误触发，同时保留
Custom23 high-turn `+8.6pp`；但它是 post-hoc safety candidate，需要新的独立转向
session 才能升级为 release-level 模型。

## 实验覆盖

- extractor inventory：YOLO-Pose high、AlphaPose、FMPose3D、MotionAGFormer、
  TCPFormer、WHAM；区分 direct、3D-derived、2D proxy 和 canonical-missing。
- orientation contract：2D π-periodic proxy、3D torso heading、direct axis-angle/
  quaternion、timestamp derivative、valid/degeneracy masks。
- physical audit：六种 extractor skeleton 与独立 IMU gyro join。
- source-target consistency：TC 与 Custom 均使用 MotionBERT/AlphaPose cache；Custom
  按 `source_person` 物化窗口。
- model ablations：no-orientation、2D proxy、3D heading、gate、concat、gyro-cross、
  conditional residual、activity auxiliary、weighted InfoNCE、25/50/100% hard
  negatives、8-bin turn-onset prediction。
- evaluation：Custom23 validation/test 无重叠；group-level high/low；5-seed frozen
  baseline；57/22/24 negative controls；作废的泄漏实验和 superseded strata 均保留记录。

## 权威产物

- 汇总：`/data/fzliang/reid-project/g12/e4_1_confirmation_summary.json`
- MoE raw：`/data/fzliang/reid-project/g12/e4_1_physical_turning_moe.json`
- frozen protocol：`/data/fzliang/reid-project/g12/e4_1_source_aligned/motionbert_alphapose_cache_v2/manifests/custom23_frozen_protocol.json`
- 详细结果：`E4.1:orientation_amplification/results/results.md`

## 决策

G12 的探索目标已达到：已证明 2D AlphaPose proxy 弱、3D-derived heading 在转向窗口
存在可用于配对的物理信号，并识别出能改善 Custom23 turning target 的模型设计。
当前推荐保留 `physical_turning_moe` 作为实验候选；带 persistence 的安全路由在获得
新的独立转向视频前只标记 provisional，不宣称对 57/22/24 有收益，也不宣称通用
跨数据集提升。
