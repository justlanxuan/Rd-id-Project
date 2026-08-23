# E1 Test：Skeleton Orientation Inventory

## 测试目标

验证 orientation inventory 不是仅凭文件名推断，且每个纳入来源的字段、shape、finite、格式和 provenance 可复核。

## 最低检查

1. 所有声明为 direct orientation 的文件真实包含 orientation 字段或四元数/轴角数组。
2. TC quaternion norm 在容差内，且关节数与 header 一致。
3. SMPL `root_orient/global_orient` shape、dtype 和帧数可读。
4. 3D-derived candidate 具备足够 pelvis/shoulder/hip joints；退化帧必须计数。
5. 2D skeleton 与 Custom cache 明确标记为 proxy/missing，不升级为 world yaw。
6. 输入不包含空数组、非有限值或无法追溯的 identity。

### Extractor-focused E1′ checks

7. YOLO-Pose high、AlphaPose、FMPose3D、MotionAGFormer、TCPFormer、WHAM 各有 algorithm/canonical artifact，且 canonical `skeleton` 为 `(T,N,17,3)`。
8. canonical artifact 不得因 `pose_world`、`root_orient` 等 raw 字段存在而被误标为 direct orientation；必须记录 raw→canonical 是否传递。
9. WHAM raw `root_orient`/`pose_world` 若存在，分类应为 `direct_orientation_raw_but_not_canonical`，直到 adapter/schema 明确传递。
10. 2D 输出（YOLO-Pose high/AlphaPose）第三维 padding 不得解释为深度；3D lifting metadata 应注明 root-center/torso-scale 坐标语义。

## 通过标准

- 每个 source 都有 `orientation_class`：`direct`、`derived`、`proxy` 或 `missing`；
- 每个字段记录 shape、坐标系状态、时间字段和 provenance；
- 失败样本被列出并停止正式 promotion；
- inventory JSON 的稳定内容 hash 可重算。
- extractor inventory validator 通过，且分类与字段丢失链路逐方法可追溯。

## 运行约束

本 E1 只允许 read-only schema inspection。不得运行训练、数据重写、模型评估或未验证的 3D 后端。
