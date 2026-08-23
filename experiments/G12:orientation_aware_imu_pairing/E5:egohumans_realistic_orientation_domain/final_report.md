# E5 Final Report：EgoHumans realistic 与 source-aligned 朝向

## 结论

E5 的 fully-aligned 实验不能证明“自己的骨架提取器朝向”带来稳定配对提升。Ego、TotalCapture、Custom 均使用 MotionBERT/AlphaPose source-aligned cache；统一 0.8 s 窗口、24 model points、session-disjoint Ego split 和 Custom23 frozen test。

- 相对最好的 E5 配置是 TC-only + O3D-derived `turning_cross`，5-seed Custom23 full `0.504`，同协议 O0 `0.484`，约 `+2.0 pp`；high `0.482` vs `0.479`，low `0.532` vs `0.491`。这属于弱且不稳定信号，Ego validation 没有提升。
- TC+Ego balanced + O3D cross 的 5-seed full 为 `0.494`，baseline `0.476`，没有稳定超过 TC-only；EH-only cross 还降低 Ego validation。
- O2D AlphaPose proxy 没有稳定收益；它只能提供 image-plane 肩/髋线 proxy，不应当解释为 world yaw。
- 因此 E5 不替换 E4.1 physical turning-MoE freeze（Custom23 full `0.534`、high `0.554`）；E5 的当前最佳训练方式只是后续研究候选。

## 协议与限制

首轮 canonical Ego/TC screen 已降级为 diagnostic，因为不是 fully-aligned extractor source。Ego 七个 canonical test session 没有对应 AlphaPose cache，canonical test 只作 domain-shift diagnostic；Ego source-side 结论目前来自 session-disjoint validation。

## Artifact

- fully-aligned O3D screen：`/data/fzliang/reid-project/g12/e5_egohumans_orientation/screen_fully_aligned/`
- fully-aligned O2D control：`/data/fzliang/reid-project/g12/e5_egohumans_orientation/screen_fully_aligned_o2d/`
- fully-aligned 5-seed confirmation：`/data/fzliang/reid-project/g12/e5_egohumans_orientation/confirmation_fully_aligned/`
- summary JSON SHA256：`1dd07189bbe90f23645b4a1326ffaf9f737a19047698d2c17430312bce97326c`
