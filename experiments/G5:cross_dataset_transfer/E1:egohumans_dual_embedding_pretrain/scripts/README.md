# E1 Scripts

TODO: implement scripts for cross-dataset dual-embedding transfer.

Planned scripts:
- `A1_prepare_egohumans_cache.sh`: build local + global cache on EgoHumans.
- `A2_train_source_local.sh`: train Model-L on EgoHumans.
- `A3_train_source_global.sh`: train Model-G on EgoHumans.
- `A4_eval_source_fusion.sh`: evaluate fusion on EgoHumans test.
- `A5_eval_target_zero_shot.sh`: zero-shot evaluate on custom test.
- `A6_train_target_finetune.sh`: fine-tune on custom.
- `A7_train_target_partial.sh`: partial fine-tune (freeze Stage1).
- `A8_aggregate_results.py`: aggregate multi-seed results.
