# E2 Test：Orientation contract

## 必测项

1. 2D shoulder axis 的角度周期为 π，且不能生成 world-yaw label；
2. 3D heading 的 up axis/cross order 改变会显式改变方向，不能静默隐藏；
3. direct axis-angle 与 quaternion 得到一致 rotation/6D，quaternion sign flip 不产生假角速度；
4. 非单调 timestamp、非有限输入、缺失关节和退化 shoulder/torso frame 必须拒绝或标记；
5. yaw rate 使用真实 timestamp，只在连续 valid segment 上求导；
6. 当前 contract 不写 canonical NPZ，也不启动 pairing training。

## 通过证据

- `tests/test_g12_orientation_contract.py`；
- `contract_manifest.json` 的 `schema_version=g12.orientation_contract.v1`；
- `py_compile` 与 NumPy smoke；
- E2 results 明确记录 `ablation_status=not_started`。
