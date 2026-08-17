# G8：完整 Session 的独立 Tracklet 历史决策

本实验只改变推理期匹配策略，不训练或微调模型。每个原始 tracker
`idx` 都是 opaque tracklet；不展开列表 ID，不推断不同 ID 的关系，不在
tracklet 之间传递状态，新 ID 从自身第一条观测初始化。状态在 session
边界清空。

评测直接读取完整 Custom session 和 `skeleton_unmerged.json`，不生成或读取
Custom segment NPZ。窗口固定为 24，步长为 16。

历史策略复现旧实验的 hard-threshold SignedVote，再按本实验要求改为逐
tracklet 决策：`decay=0`、`sigmoid(3 × margin)`、阈值 `0.7`、低置信度
`preserve`、greedy assignment。每份结果同时保存不使用历史的即时
Hungarian baseline。

`configs/` 中四个 E28 配置分别只评测对应 held-out session，并加载该 fold
的 `best.pt`，避免训练泄漏。运行示例：

```bash
./run_pipeline.py \
  --config 'experiments/G8:full_session_tracklet_history/configs/e28_fold1.yaml' \
  --stages test
```
