# E1 Progress：Skeleton Orientation Inventory

## 状态

已完成只读候选路径审计、代表性 schema 抽样和两份 inventory artifact；尚未进入 E2 orientation contract。此前 source-orientation inventory 已完成，但经用户反馈确认其不是主问题；extractor-focused inventory（E1′）现作为主线。

## 已观察事实

- TotalCapture Vicon `gt_skel_gbl_ori.txt` 是 21×4 的单位四元数序列。
- TotalCapture SMPL-X 包含 `root_orient (T,3)` 轴角字段。
- EgoHumans full SMPL 每帧包含 `global_orient`；当前 Re-ID canonical cache 未传递该字段。
- 当前 Custom corrected G11 cache 只有 2D skeleton，没有可靠世界朝向。

## E1 完成记录

- 生成 `/data/fzliang/reid-project/g12/e1_inventory/orientation_inventory.json` 和 `orientation_inventory.md`。
- manifest schema：`g12.orientation_inventory.v1`。
- manifest SHA256：`3804ab8faa800507268b72f42260b3a713445f0677e6e46d2634b8ec9a21df93`。
- inventory 包含 9 个 source records；所有抽样失败数为 0。
- TotalCapture Vicon：1 个 raw orientation 文件 + 10 个 orientation archives，archive 内 92 个 orientation members；抽查四元数为 21 joints × 4，norm `0.9999991–1.0000009`。
- TotalCapture SMPL-X：37 个 NPZ，抽样 `root_orient` 均为 finite `T×3` axis-angle。
- EgoHumans fitted SMPL：70,113 个 frame files，抽样均含 finite `global_orient (3,)`；标记为 estimated candidate。
- fzliang canonical TotalCapture 仅保留 3D skeleton、归为 derived；EgoHumans canonical 归为 2D proxy；Custom corrected cache 归为 orientation-missing。
- E1 validator 通过：manifest hash、类别、finite、TC quaternion norm、direct/derived/proxy/missing 分类均验证。

## E1′：Extractor-focused correction

- 审计 `/data/lyxie/ReID/Pipeline/Skeleton_Extractors`、`S06_source_ablation`、`RB-Skeleton-Aug/S06_Algo_Aug/algorithm_outputs` 和 YOLO-Pose high/WHAM raw artifacts。
- 六种实际 extractor 均有 88 个 algorithm-output NPZ 与 304 个 S06 canonical NPZ；canonical schema 只有 `skeleton`/`extract_skeleton (T,N,17,3)`、visibility、identity/alignment metadata，没有 rotation/root orientation 字段。
- YOLO-Pose high、AlphaPose：2D image-plane proxy；其 z=0 padding 不是深度/世界坐标。
- FMPose3D、MotionAGFormer、TCPFormer：3D root-centered torso-scaled H36M17 xyz，可派生坐标系依赖的 torso heading，但无直接 rotation。
- WHAM raw processed artifacts 含 `root_orient (T,3)`、`pose_world` 等直接朝向候选；当前 WHAM canonical adapter 只导出 17 joints，因此分类为 `direct_orientation_raw_but_not_canonical`。
- Extractor inventory JSON：`/data/fzliang/reid-project/g12/e1_inventory/extractor_orientation_inventory.json`。
- Extractor validator 通过：`PASS extractor inventory: 6 methods`。

## 下一步

1. E2 contract 已冻结于 `E2:orientation_contract/`，固定 2D proxy、3D-derived heading、WHAM raw root orientation 三条轨道；
2. E2 synthetic contract tests、关节索引/坐标语义和退化 mask 审计已通过；
3. 若传递 WHAM `root_orient`，新建 adapter/schema version，不能直接复用当前 S06 canonical 文件；
4. 进入 E3 physical orientation–gyro audit；E4 pairing ablation 仍未开始。
