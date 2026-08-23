# E5 Progress：EgoHumans realistic orientation/domain

## 2026-08-22

- 按 HAROS 将请求路由为 G12 内独立 E5；E4.1 保持只读。
- 只读盘点完成：EgoHumans realistic canonical sequence 含 3D `gt_skeleton/gt_skeleton_meters`、`gt_visibility` 和 7D IMU；native rate=20 Hz。
- 确认正式 0.8 秒 Ego 窗口必须使用 16 native frames，再重采样为 24 model points；旧 24-frame manifest 标记为 superseded。
- 确认 O2D（AlphaPose proxy）与 O3D（3D joint-derived heading）必须分轨；SMPL `global_orient` 只作 reference。
- 当前阶段：准备生成 session-level split 与运行门禁，尚未训练或生成性能结论。

## Screen audit and source-aligned rerun (2026-08-22)

- 首轮 canonical-Ego screen 已完成 26 个 seed-run；它使用 canonical 3D skeleton，仅保留为协议/训练稳定性诊断，不作为 extractor-orientation 结论。
- 发现问题后已切换至同一套 S06 AlphaPose source-aligned cache，使用 `e5_ego_train.csv` 与 `e5_ego_validation.csv`，并将重跑结果写入独立的 `screen_source_aligned/`。
- Ego 的七个 canonical test session 尚无对应 source-aligned cache，因此 canonical test 只作为标记清楚的 domain-shift diagnostic，不参与 promotion。

## Screen findings (2026-08-22)

- 已修正 TC source provenance：前一轮 canonical-TC 结果降级为诊断；最终 fully-aligned screen 使用 Ego/TC/Custom 的 MotionBERT/AlphaPose source-aligned cache，共完成 O3D/O3D-rate 与 O2D proxy 矩阵。
- fully-aligned TC-only O3D `turning_cross` 5-seed 相对 O0 的 Custom23 high/full/low 为 `0.482/0.504/0.532` vs `0.479/0.484/0.491`，增益仅 `+0.4/+2.0/+4.1 pp`，且 Ego validation 没有提升。
- fully-aligned TC+EH balanced 5-seed full 为 `0.494` vs baseline `0.476`，方向不稳定；EH-only cross 也损害 Ego validation，故没有稳定的共同训练晋级。
- O2D AlphaPose proxy 没有稳定收益；当前证据不支持把 2D 肩/髋线当成可迁移的朝向配对信号。
- E5 汇总写入 `/data/fzliang/reid-project/g12/e5_egohumans_orientation/e5_screen_summary.json`；E4.1 freeze 未被覆盖。
