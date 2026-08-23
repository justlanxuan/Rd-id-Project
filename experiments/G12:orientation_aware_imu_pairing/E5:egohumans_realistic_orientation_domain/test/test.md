# E5 Test Contract：EgoHumans realistic orientation/domain

## 测试目标

验证 E5 的数据、时间、朝向 provenance、source regime 和 frozen Custom23 评价没有泄漏或窗口失配。

## 必须通过的门禁

1. Ego train/validation/test 按 session 不重叠；重叠窗口不得跨 split。
2. Ego native windows 为 16 frames@20Hz=0.8s；TC 为 48@60Hz；Custom 为 24@30Hz；所有 model input 为 24 points。
3. `gt_skeleton`/`gt_skeleton_meters`、AlphaPose proxy、MotionBERT 3D-derived heading 和 direct `global_orient` provenance 分轨记录。
4. O3D heading 只使用左右肩、pelvis、thorax 和真实 timestamp；heading rate、validity、degeneracy 可审计。
5. TC+EH-balanced sampler 按 group 而非 raw row 平衡；O0 模型不能读取 orientation tensor。
6. Custom23 使用 frozen protocol，阈值/epoch 只由 validation 选择，test 只运行一次；57/22/24 不参与调参。
7. 每个 run 保存 config、manifest hash、checkpoint hash、raw `correct/total`、FrameAcc、margin 和 seed。

## 复现入口

```bash
PYTHONPATH=. /home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G12:orientation_aware_imu_pairing/E5:egohumans_realistic_orientation_domain/scripts/E5_build_session_split.py
```

Manifest/orientation gate：

```bash
PYTHONPATH=. /home/fzliang/miniconda3/envs/reid_project/bin/python \
  experiments/G12:orientation_aware_imu_pairing/E5:egohumans_realistic_orientation_domain/test/scripts/validate_e5_manifests.py \
  --split /data/fzliang/reid-project/g12/e5_egohumans_orientation/manifests/egohumans_e5_session_split.json \
  --aligned-train /data/fzliang/reid-project/g12/e4_1_source_aligned/motionbert_alphapose_cache_v2/manifests/e5_ego_train.csv
```

Fully-aligned screen/confirmation entries are `scripts/E5_run_screen.py`,
`scripts/E5_run_o2d_screen.py` and the recorded run directories under
`/data/fzliang/reid-project/g12/e5_egohumans_orientation/`. The initial
canonical-Ego and canonical-TC screens are explicitly diagnostic because they
did not use the same extractor cache; they are not merged into the fully-aligned
result table.

## 失败判定

- 任一 split session overlap、native duration mismatch、orientation shape/provenance mismatch、candidate leakage 或 frozen test 事后调参，整组结果降级为 diagnostic，不得进入主表。
