# E3 Progress

- 2026-08-21：完成只读审计脚本，覆盖 TotalCapture canonical 3D heading control、WHAM raw `pose_world` direct orientation、EgoHumans 四人 matched/shuffled-person controls，以及六种 S06 extractor 的可用性 null scan。
- 2026-08-21：生成 `/data/fzliang/reid-project/g12/e3_physical_audit/orientation_gyro_audit.json`；没有运行 pairing ablation。
- 2026-08-21：人类指出首轮把“骨架文件不含 gyro”误作阻塞条件。已纠正为 skeleton orientation × external IMU gyro join；旧 extractor-missing-gyro 解释 superseded。
- 2026-08-21：全量连接六种 algorithm outputs：88 sequences、313 person tracks/method、1,878 records、failures=0。corrected artifact 为 `/data/fzliang/reid-project/g12/e3_physical_audit/extractor_orientation_imu_join_corrected.json`。
