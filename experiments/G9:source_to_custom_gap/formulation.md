# G9：Source-to-Custom 骨架与跨模态域差异分解

## Need

G6 已经完成 TotalCapture、EgoHumans 到 Custom 的统一 Re-ID 基准，但结果只能说明存在跨域性能差异，不能说明差异来自哪里。当前结果中，EgoHumans→Custom 的 zero-shot micro FrameAcc 为 62.02%，fine-tune 后为 34.46%；TotalCapture→Custom 分别为 54.06% 和 50.92%。这说明“Custom 更难”不是充分解释，必须区分输入分布、跨模态配准、动作组成、跟踪身份和评测协议等因素。

项目历史上存在多种骨架源：TotalCapture Vicon/GT、EgoHumans pose2d、AlphaPose、YOLO-Pose high、FMPose3D、MotionAGFormer、TCPFormer 和 WHAM。不同来源可能同时改变坐标空间、2D/3D 表示、噪声、遮挡、置信度和 tracklet 语义。未经内容指纹和真实 smoke 验证的 PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 只作为候选，不进入第一版正式结论。

## Goal

建立一个可复现的 gap decomposition 框架，回答：

1. gap 主要来自 IMU、骨架、IMU-骨架关系，还是评测协议；
2. 不同骨架源对 source→Custom transfer 的影响是否一致；
3. 哪些动作复杂度、遮挡和 tracker 条件最容易失败；
4. 为什么部分 fine-tune 条件反而损害 zero-shot 表征；
5. 哪些可控干预（归一化、时间对齐、分支适配）能够修复性能。

## Hypotheses

- **H1：IMU sensor gap**：位置、坐标系、采样、单位或噪声差异导致 IMU 分布和物理语义不一致。
- **H2：Skeleton measurement gap**：GT/mocap、2D detector、3D lifter 和 SMPL/HMR 的质量差异导致 joint/bone/置信度分布不同。
- **H3：Cross-modal alignment gap**：IMU 与 skeleton 的时间延迟、person mapping 或运动相关关系在 Custom 中改变。
- **H4：Motion-complexity gap**：Custom 的动作速度、jerk、谱熵、交互和动作转换分布不同，导致模型在长尾复杂动作上失效。
- **H5：Tracking/identity gap**：遮挡、track fragmentation、ID switch 和 candidate-group 结构改变了 FrameAcc 难度。
- **H6：Protocol/adaptation gap**：窗口、stride、normalization、inner validation 或 fine-tune 过程引入了非算法性的差异。

## Scope

正式分析固定 G6 的数据协议：`window_len=24`、`stride=16`、四个 Custom held-out sessions、seeds `0/42/123`。G9 使用独立的协议记录和结果目录，不修改或覆盖 G6 结果。

正式骨架源分为三类：

| 类别 | 骨架源 | 第一版状态 |
|---|---|---|
| Reference | TotalCapture Vicon/GT | 纳入 |
| 2D | EgoHumans pose2d、AlphaPose、YOLO-Pose high | 纳入 |
| 3D | FMPose3D、MotionAGFormer、TCPFormer | 纳入，先验证内容独立性 |
| SMPL/mesh | WHAM | 纳入，保留原始与统一格式 |
| Candidate | PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper | 仅在真实 smoke 通过后追加 |

## Primary metrics

- FrameAcc：保存 `correct/total`、micro/weighted、macro-session 和逐 session 结果；
- source→Custom 的 zero-shot、fine-tune、direct 对照；
- 每关节/每 IMU 通道的 Wasserstein、MMD、CORAL 距离；
- IMU-skeleton cross-correlation lag、CCA/HSIC 或等价跨模态对齐指标；
- motion energy、jerk、谱熵、活动关节率、遮挡率、tracklet 长度和 candidate-group size 分层结果；
- seed/session 离散度，以及干预前后的配对差值。

## Validity constraints

- 2D、3D 和 SMPL 输入分轨比较，不把不同坐标空间直接排名；
- 所有骨架源必须有 provenance、内容 hash、joint mapping 和 finite/schema 验证；
- 文件数量相同不能证明算法独立，必须检查张量内容和相关性；
- held-out Custom session 不得参与 normalization、checkpoint selection 或超参数选择；
- G8 的历史 tracklet 策略只作为诊断变量，不作为默认改进；
- 未通过真实 smoke 的候选后端不得进入正式表；
- 所有正式结果从机器可读 raw counts 自动汇总，不能手抄最终百分比。
