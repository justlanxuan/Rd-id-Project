# E2 Progress：Extractor orientation contract

## 已完成

- 冻结 `g12.orientation_contract.v1`，将 2D proxy、3D-derived heading、direct raw orientation 和 orientation-missing 分轨。
- 新增纯 NumPy 实现 `src/features/orientation.py`，不修改 canonical skeleton schema。
- 明确 H36M17 joint order、2D shoulder-line π 周期、3D shoulder/up/cross 规则、direct axis-angle/quaternion/6D 规则。
- 明确严格 timestamp、连续 valid segment、rate_valid、degeneracy_reason 和 raw provenance 要求。
- 新增 `tests/test_g12_orientation_contract.py` 与 E2 contract manifest/results/test 文档。

## 验证

- `/usr/bin/python3.10` NumPy smoke：2D、3D、axis-angle、quaternion sign continuity 均通过；`py_compile` 通过。
- base shell 没有 pytest，但 `reid_project` Python 3.10 环境已运行 pytest：G12 contract + G10 feature focused tests `18 passed`；Ruff focused check 通过。
- 未修改正式 preprocess/schema，未生成 orientation-aware pairing 性能结果。

## 下一步

进入 E3：先在经过 E1/E2 的 extractor artifact 上做 heading/gyro physical audit，按 source、sensor placement、coordinate frame 和 motion stratum 分层；E3 通过前不做 E4 ablation。
