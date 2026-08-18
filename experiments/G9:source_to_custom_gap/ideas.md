# G9 Ideas：Source-to-Custom Gap

## 已批准的分析方向

1. IMU raw/normalized 分布、频谱、jerk、坐标和传感器位置审计。
2. 2D、3D、SMPL 三条 skeleton representation track 分开比较。
3. IMU-only、skeleton-only、fusion 三种模态消融。
4. IMU-skeleton lag、cross-correlation、CCA/HSIC 和 person mapping 审计。
5. motion energy、谱熵、活动关节率、动作转折和交互复杂度分层。
6. confidence、遮挡、tracklet 长度、ID switch、candidate-group size 分层。
7. source normalization、target normalization、frozen adapter、branch-wise fine-tune 和 full fine-tune 对照。
8. 对 FMPose3D、MotionAGFormer、TCPFormer 等目录相同覆盖率结果执行内容指纹检查，避免重复产物伪装成算法差异。

## 暂不纳入默认路径

PromptHMR、Human3R、GENMO、SMPLest-X、TRAM、VIBE、DenseWarper 只有在完成依赖、权重、真实非空输出和 canonical schema smoke 后，才可作为追加骨架源。
