# E7 Results：最终 Gap Manifest

已生成 `/data/fzliang/reid-project/g9/g9_final_gap_manifest.json`。

当前状态为 `diagnostic_complete_protocol_boundaries_explicit`：E1/E2/E4/E5 的可追溯 screening、S06 坐标控制、Custom IMU quaternion 控制、S06 prediction strata 和 Custom detector-ID audit 已齐备。Manifest 仍禁止把固定检查点干预外推为重训后的全域因果归因，并把 full-xyz 与 S06 独立 ID 标为协议边界。它汇总：

- per-source selective gate 和最小可信子集；
- H36M17、2D/3D representation、坐标空间、person/IMU/time join 证据；
- YOLO 有限极值、Custom invalid quaternion、重复产物检查；
- IMU schema/unit path、lag、运动复杂度、visibility/tracklet 结果；
- 现有 G6 `correct/total` 与已完成的 S06 六源×raw/screen fixed-checkpoint source-sweep 控制条件。
- 现有 G6 105 个 session clips 的 prediction-level complexity/candidate-group/visible-people 分层；该分层不替代尚未重训的 S06 source-domain benchmark。
- S06 528 个逐序列预测与 motion/visibility/fragmentation-proxy 的 correct/total 分层；fragmentation 不等于 ID-switch。
- 四个 Custom held-out session 的 raw、invalid-fill-only、unit-normalized IMU 对照及逐 session 分母。
- Custom AlphaPose raw detector `idx` 与 GT bbox 的 transition audit；S06 无独立 detector ID 的限制。

若后续研究需要 full-xyz 或 S06 ID-switch 归因，必须新建 xyz-compatible encoder 或保留独立 detector IDs；这两项不被当前 G6 协议隐式替代。
