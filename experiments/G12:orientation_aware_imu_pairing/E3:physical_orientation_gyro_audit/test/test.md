# E3 Test Contract

- `run_e3_audit.py` 必须产生 `g12.e3.physical_orientation_gyro_audit.v1` JSON。
- 结果必须同时含 zero-lag、±1 s lag screen、motion strata、time-shuffle null；EgoHumans 必须含 matched 与 shuffled-person records。
- 六种 S06 extractor 必须分别从 algorithm output 读取朝向，并与 external realistic-IMU 按 sequence/person/frame 连接；不得要求 skeleton 文件自带 gyro。
- 每种方法必须覆盖 88 sequences、313 tracks；全量 join failures 必须为 0。
- matched median 必须与相同 estimator 的 shuffled-person 和 time-shuffle controls 一起报告，禁止只报告三轴最大相关。
- gyro 单位固定为 `rad/s`，orientation rate 固定为 `rad/s`；不得从 legacy 48D quaternion 静默伪造独立 gyro。
- 审计不得写入模型 checkpoint、正式 preprocess schema，或运行 pairing ablation。
