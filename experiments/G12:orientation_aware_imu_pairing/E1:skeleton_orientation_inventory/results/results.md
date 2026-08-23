# E1 Results：Skeleton Orientation Inventory

## 状态

E1 inventory 已完成（read-only）。机器可读 artifact：

`/data/fzliang/reid-project/g12/e1_inventory/orientation_inventory.json`

Manifest SHA256：`3804ab8faa800507268b72f42260b3a713445f0677e6e46d2634b8ec9a21df93`

Markdown 展开版见同目录的 `orientation_inventory.md`。该 artifact 只记录 schema/provenance，不包含任何训练或性能结论。

## 预审摘要

| Source | Orientation class | Evidence | Promotion status |
|---|---|---|---|
| TotalCapture Vicon | direct | `gt_skel_gbl_ori.txt` 21 joint quaternions | candidate |
| TotalCapture SMPL-X | direct | `root_orient (T,3)` axis-angle | candidate; convention audit required |
| EgoHumans fitted SMPL | direct/estimated | per-frame `global_orient` | candidate; estimate/coordinate audit required |
| EgoHumans 3D pose | derived | `pose3d`, `fit_poses3d`, `refine_poses3d` | derived-only |
| fzliang TotalCapture canonical | derived | `gt_skeleton_meters` | derived-only; original ori dropped |
| fzliang EgoHumans canonical | proxy/missing | 2D `gt_skeleton` | no world-yaw promotion |
| fzliang Custom corrected cache | missing/proxy | 2D `skeleton`/`extract_skeleton` | no formal orientation target |

## Inventory 结果

| Source | Class | Status | 发现规模 | 抽样 | 失败 |
|---|---|---|---:|---:|---:|
| TotalCapture Vicon | direct | candidate | 1 raw + 10 archives / 92 orientation members | 1 | 0 |
| TotalCapture SMPL-X | direct | candidate | 37 NPZ | 3 | 0 |
| EgoHumans fitted SMPL | direct | candidate_estimated | 70,113 frame files | 3 | 0 |
| EgoHumans extracted pose3d | derived | derived_only | 456 files | 3 | 0 |
| EgoHumans fit/refine pose3d | derived | derived_only | 70,113 / 70,113 files | 3 + 3 | 0 |
| fzliang TotalCapture canonical | derived | derived_only | 138 sequence NPZ | 3 | 0 |
| fzliang EgoHumans canonical | proxy | orientation_missing | 114,192 sequence NPZ | 3 | 0 |
| fzliang Custom canonical | missing | orientation_missing | 2,209 sequence NPZ | 3 | 0 |

关键验证：

- 抽查的 TotalCapture Vicon 文件为 21 关节、`quaternion_wxyz`，quaternion norm 范围 `0.9999991–1.0000009`，全部 finite。
- TotalCapture SMPL-X `root_orient` 抽样均为 finite `T×3` 轴角，采样文件声明 `mocap_frame_rate=60`。
- EgoHumans fitted SMPL 抽样均含 finite `global_orient (3,)`，但属于估计产物，不是光学 GT。
- 当前 canonical EgoHumans/Custom 样本未发现 orientation 字段；分别归类为 `proxy` 与 `missing`，不升级为 world yaw。
- 稳定 fingerprint 只依赖文件内容和相对路径；不纳入绝对路径、mtime 或目录名猜测。

## Gate 结论

E1 的 schema/provenance inventory gate 通过；E2 仍需完成坐标系、quaternion sign continuity、axis-angle conversion、yaw wrap 和真实 timestamp derivative 契约。在 E2 通过前，任何 direct orientation 只可作为候选 source，不能进入正式配对训练。
