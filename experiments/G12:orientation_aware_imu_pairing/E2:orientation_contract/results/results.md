# E2 Results：Extractor orientation contract

## 状态

契约实现与 synthetic/edge-case 验证已完成；尚未修改正式 preprocess/schema，尚未运行 orientation-aware pairing ablation。

实现入口：[src/features/orientation.py](../../../../src/features/orientation.py)。

## 冻结内容

- `derive_2d_torso_proxy`：肩线的 π-periodic image-plane proxy；输出 `sin(2θ), cos(2θ)`，不宣称 world yaw。
- `derive_3d_torso_heading`：H36M17 3D joints 的 shoulder lateral/up/cross heading；up axis 与 cross order 必须显式写入 provenance。
- `direct_root_orientation`：axis-angle/quaternion → rotation matrix、6D、yaw、yaw rate；quaternion 做 normalization 和 sign continuity。
- 所有轨道输出 `orientation_valid`、`rate_valid`、`degeneracy_reason`、`coordinate_frame` 和 `orientation_source`。
- timestamp 必须严格递增秒值，导数不跨越 invalid segment。

机器可读契约：[contract_manifest.json](contract_manifest.json)。

## 证据

`tests/test_g12_orientation_contract.py` 覆盖：2D π-periodicity、真实 timestamp rate、missing/degenerate mask、3D cross-order flip、axis-angle/6D、quaternion sign continuity、非单调 timestamp 和 2D padded-z 拒绝。

当前没有任何配对准确率或 IMU 相关性结论；E3 才开始 physical audit。
