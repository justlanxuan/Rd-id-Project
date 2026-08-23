# E1 Plan：Skeleton Orientation Inventory

## 目标

只读盘点 `/data/lyxie` 与 `/data/fzliang` 中我们自己的 extractor artifacts 的 orientation 字段、格式、坐标语义、时间覆盖和 provenance，为 G12 orientation contract 提供事实基础。数据集原生 orientation 仅保留为 reference/control。

## 纳入范围

- YOLO-Pose high 与 AlphaPose 原始/统一 2D skeleton；
- FMPose3D、MotionAGFormer、TCPFormer 的 3D lifting skeleton；
- WHAM canonical 17-joint skeleton 与 raw `root_orient`/`pose_world` artifact；
- S06 source-ablation 和 S06 canonical adapter 的字段传递/丢失；
- TotalCapture Vicon/SMPL-X、EgoHumans fitted SMPL 作为 reference/control。

## 不做

- 不修改正式 preprocess/schema；
- 不运行训练、评测或第三方 3D 后端；
- 不把 2D shoulder/hip axis 写成世界 yaw；
- 不生成可进入 G10/G11 主表的性能结果。
- 不把 raw orientation 字段误报为 canonical pair input 已经可用。

## 产出

- `results/orientation_inventory.md`；
- `results/extractor_orientation_inventory.md`；
- `results/orientation_inventory.json`（大型内容写入 `/data/fzliang/reid-project/g12/e1_inventory/`）；
- extractor inventory JSON（大型内容写入 `/data/fzliang/reid-project/g12/e1_inventory/`）；
- `test/test.md` 与必要的 read-only inventory script；
- 每个 extractor 的 direct-propagated/raw-only/3D-derived/2D-proxy 分类和失败原因。
