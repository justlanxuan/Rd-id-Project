# E7 Results：最终 Gap Manifest

已生成 `/data/fzliang/reid-project/g9/g9_final_gap_manifest.json`。

当前状态为 `diagnostic_complete_causal_controls_pending`：E1/E2/E4/E5 的可追溯 screening 已齐备，但 manifest 明确禁止把观察性相关性当成因果归因。它汇总：

- per-source selective gate 和最小可信子集；
- H36M17、2D/3D representation、坐标空间、person/IMU/time join 证据；
- YOLO 有限极值、Custom invalid quaternion、重复产物检查；
- IMU schema/unit path、lag、运动复杂度、visibility/tracklet 结果；
- 现有 G6 `correct/total` 与尚未运行的 S06 source-sweep 控制条件。

必须继续完成的控制实验已经写在 `next_required_controls`，包括 representation-separated source sweep、统一 7D IMU 的 filtered/unfiltered fusion、complexity/tracklet prediction 分层以及 raw detector track-ID 审计。
