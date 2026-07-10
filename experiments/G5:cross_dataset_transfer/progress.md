# G5 Progress: Cross-Dataset Transfer

## 当前状态

- [x] G5 goal 已初始化，拆分为 E1（EgoHumans 源）与 E2（realistic IMU 源）。
- [x] E1：完成 EgoHumans cache 构建、source Model-L/Model-G 训练、zero-shot 与 fine-tune（seed0）。
- [x] E2：完成 realistic w24 cache 构建。
- [x] E2：完成 6 seeds 的 source Model-L / Model-G 训练。
- [x] E2：完成 6 seeds 的 custom zero-shot 评估。
- [x] E2：完成 6 seeds 的 custom target fine-tune（local / global）。
- [x] E2：完成 6 seeds 的 fine-tuned fusion 评估（none / zscore / minmax）。
- [x] E2：完成结果聚合与 `results.md`。
- [x] E0：完成与 E1 / E2 / E4b / G3 / G4 的统一对比表。
- [x] E3：完成 realistic single-source low-LR fine-tune，6 seeds `0.7413 ± 0.0241`。
- [x] E4：完成 realistic dual low-LR seed123 诊断，确认只能部分修复 E2 collapse，暂不扩展。
- [x] E5：完成 adaptive gate oracle 诊断，确认 alpha selection 不能救当前 E2 transfer dual。
- [x] E6：完成 IMU format audit，确认 E3/E4b 20 Hz 7D path 是当前最合理 transfer 主线，TotalCapture 不宜直接混入。
- [x] E9：完成 explicit LeftWrist source/custom cache、realistic LeftWrist source pretrain -> custom zero-shot/fine-tune，以及 direct custom LeftWrist from-scratch，matched seeds 0/42/123。
- [x] E10-A1：完成 skeleton differential / relative motion alignment diagnostic 与 filter sweep。
- [x] E11-A1：完成 shoulder-anchored traditional kinematic matching sanity check。
- [x] E11-A2/A3：完成 50D shoulder-anchored kinematic cache 与 MoBind feature-path direct seed0；结果为负，停止该接口。
- [x] E11-A4/A5/A6：完成 learned lag-corr、temporal assignment、selector headroom analysis；clip-global kinematic assignment 达到 `0.6831`，oracle selector 达到 `0.8914`。
- [x] E11-A7/A8/A9/A10：完成 selector diagnostics；A10 hybrid signed-window / abs-global consistency policy 达到 `0.9014`，首次超过 raw-keypoint direct seed0 `0.8396`。
- [x] E11-A11：完成 frozen A10 rule 的 multi-seed/bootstrap validation；A10 `0.9014` 超过 direct custom seeds 0/42/123 和 3-seed mean `0.6346 ± 0.1681`。
- [x] E12-A1：完成 realistic source-trained kinematic selector 多 source-fold 验证；6 folds 全部选择 `abs_global`，custom 为 `0.6831 ± 0.0000`。
- [x] E13-A1：完成 physics-strengthened rule distillation 调研、brainstorming 和下一阶段 HAROS 计划。
- [x] E13-A2/A3：完成 clip-level teacher signals 与 LOOCV model+rule selector；A3 达到 `0.6998`，仅弱超 E12 `0.6831`，不足以完成目标。
- [x] E13-A4/A4b：完成 MLP lag-corr pair scorer multi-seed；mean train accuracy `0.9888` 但 eval `0.5989 +/- 0.0127` over seeds `0/1/2/3/42/123`，负结果确认。
- [x] E13-A5：完成 window-level temporal selector；no-leak clip-CV `0.9014`，更严格 session-CV `0.8251`，首次在 held-out protocol 下超过 G4/E11 dual fusion mean `0.7516`。
- [x] E13-A6：完成 GPU-trained MLP window temporal selector 多 seed；clip-CV `0.5908 +/- 0.0450`，session-CV `0.5898 +/- 0.0068`，负结果。
- [x] E14-A1/A1b/A2/A3/A4/A5/A6：完成 shoulder-local vector idea 的首轮验证。稳定 vector 表征降低 jitter 但降低 IMU corr；无增强 pair model seed0 eval `0.3510`。Session-heldout A3 为 `0.5026 +/- 0.1250`，仍不稳。原划分增强版 A4 达到 `0.9005 +/- 0.0119`，但泄露审计发现 `309/455` eval windows 与 train windows 重叠；非泄露增强版 A5 只有 `0.4359 +/- 0.1072`。Protocol A 审计 A6 显示当前 converted custom data 中没有非 evaluation clips 可训练。
- [x] E15-A1/A2：完成 Protocol A data-integration step。原始 full-session tail 没有可用 100-frame non-eval windows；外部 supervised custom NPZ 已转换为 E9-style cache，共 4589 windows（train 3863 / val 726）。注意 A2 是 `L_LowArm -> 7D`，不是 true LeftWrist placement。
- [x] E16：完成 MoBind best-config one-session-out 4fold，seed0 mean `0.5998 +/- 0.2297`。
- [x] E17-A1/A2/A3/A4：完成 topology-aware arm-vector model preflight、seed0 one-session-out 4fold 训练评估、targeted ablation 与汇总；full A2 为负结果，`0.3789 +/- 0.1760`，best ablation `raw_imu=0.5087 +/- 0.1215`。
- [x] E18：完成 source-only realistic EgoHumans LeftWrist pretrain checkpoints 在 custom one-session-out protocol 下的 zero-shot 评估；3 source seeds x 4 folds = `0.5009 +/- 0.2476`，低于 E16 direct custom `0.5998 +/- 0.2297`。
- [x] E19：完成 anti-shortcut protocol 源码修正。新增 G5 共享 split audit / label-shuffle / temporal-shift / static-control 工具，并把 E17 训练/复评入口接入；E16 四个 session-out fold audit 全部通过。
- [x] E20：完成非泄露 one-session-out 多 seed 最终对比。MoBind direct custom `0.4947 +/- 0.2326`，MoBind realistic transfer `0.4504 +/- 0.1780`，E17 realistic transfer `0.4685 +/- 0.2287`，均为 4 folds x 3 seeds。
- [x] E21：完成 A0-A8。Hybrid raw-pose + shoulder-vector source pretrain -> custom session-out transfer 从 3 seeds 扩到 6 seeds 后达到 `0.6550 +/- 0.1823` over 24 fold-runs。新 seeds 1/2/3 单独为 `0.6840 +/- 0.1718`。同架构 direct custom/no source pretrain 为 `0.5120 +/- 0.1987`。全量 label-shuffle control 为 `0.4628 +/- 0.1429`，aligned-shuffle paired delta `+0.1923`，CI `[+0.1123, +0.2769]`，支持 E21 不是纯捷径。
- [~] E22：完成 A1 spectral/audio-style diagnostic；spectral-only `0.5316`、hybrid `0.5443` 均低于 time-domain `0.6709`，暂不启动 heavy spectral neural branch。

## 最近更新

- 2026-07-03: 启动 E1，构建 EgoHumans cache，开始 source Model-L / Model-G seed0 训练。
- 2026-07-03: E2 realistic w24 cache 构建完成（21,179 entries）。
- 2026-07-03: E2 完成 6 seeds 的 source 训练与 zero-shot 评估。
- 2026-07-03: E2 完成 6 seeds 的 target fine-tune 与融合评估，结果写入 `E2:realistic_dual_embedding_pretrain/results/results.md`。
- 2026-07-05: 根据 E1/E2/E4b/G4 结果更新 G5 总体计划：下一步优先 E3 conservative fine-tune、E4 adaptive local/global gate、E5 IMU canonicalization/domain randomization。
- 2026-07-05: 完成 E0 cross-dataset comparison，结果写入 `results/cross_dataset_comparison.md/json`。
- 2026-07-05: 完成 E3-A4 low-LR full fine-tune 6 seeds：`0.7413 ± 0.0241`；在 2026-07-06 wrist-side 修正后，该结果改为 historical mislabeled-wrist baseline。
- 2026-07-05: 完成 E4 realistic dual conservative fine-tune seed123 诊断：最佳 `0.4894`，只部分修复 E2 seed123，不值得扩到 6 seeds。
- 2026-07-05: 完成 E5 adaptive gate oracle 诊断：G4 direct w24 frame oracle `0.8494 ± 0.1009`，但 G5/E2 fine-tune minmax clip oracle 仅 `0.6877 ± 0.1417`，低于 E3。
- 2026-07-05: 完成 E6 IMU format audit：E4b/E3 已有 20 Hz custom RightWrist-named 7D cache；在 2026-07-06 wrist-side 修正后，该 cache 应按 LeftWrist 语义理解。TotalCapture 48D sensor placement 不匹配左手腕单 IMU。
- 2026-07-05: 完成 E6 target train-only acc-jitter ablation（seeds 0/42/123）：结果与 E3 low-LR 完全持平，mean delta `-0.0000`，不扩展该增强强度。
- 2026-07-06: 修正关键事实：custom IMU 实际佩戴在 LeftWrist。E3/E4/E5/E6 的 RightWrist motion target 结果改为 historical mislabeled-wrist baselines。
- 2026-07-06: 新建 E9 LeftWrist revalidation，完成 realistic LeftWrist source cache（4659 windows）与 custom LeftWrist alias cache（316 windows）；A2/A3/A3b/A4/A6/A7/A8/A9/A10 脚本已准备。
- 2026-07-06: E9 direct custom LeftWrist from-scratch 已完成 seeds 0/42/123：`0.8396 / 0.6364 / 0.4279`，mean `0.6346 ± 0.1681`。
- 2026-07-06: E9 transfer seed0 completed：zero-shot `0.7244`，low-LR fine-tune `0.7270`。
- 2026-07-07: E9 transfer seeds 42/123 completed through detached tmux. Corrected LeftWrist 3-seed results: zero-shot `0.6115 ± 0.1409`，low-LR fine-tune `0.6148 ± 0.1638`，direct custom `0.6346 ± 0.1681`。
- 2026-07-07: E10-A1 completed through tmux. Skeleton-IMU alignment diagnostic shows filtered bone geometry/relative motion has signal: top best-lag corr improves from kernel1 `0.3613` to kernel9 `0.4341`; pure differential replacement is not yet justified.
- 2026-07-07: E10-A1b added explicit elbow included-angle differential. Elbow angle is a strong feature (`0.4276` best-lag corr under smooth9), and elbow angular velocity has high window-level energy correlation with IMU dynamics (`0.7050`).
- 2026-07-07: E10-A3 direct custom seed0 replacement-feature A/B completed through tmux. Results are negative versus E9 raw-keypoint direct seed0 `0.8396`: `hybrid_v1=0.5330`, `geometry_only=0.4445`, `dynamics_only=0.4387`. Do not expand these low-dimensional replacement features.
- 2026-07-07: E11-A1 traditional shoulder-anchored kinematic matcher completed through tmux. No-training matcher reaches `0.6352`, beating E10-A3 learned `hybrid_v1` (`0.5330`) and supporting the hypothesis that E10-A3 failed due to representation/model-interface design.
- 2026-07-07: E11-A2/A3 completed through tmux. Built 50D `shoulder_kinematic_v1` cache, but direct custom seed0 through MoBind `motion_type: feature` reached only `0.3355`. This confirms the next step must be a purpose-built lag/cross-attention matcher or raw-pose + kinematic auxiliary branch, not another feature-only replacement.
- 2026-07-07: E11-A4/A5/A6 completed through tmux. Learned lag-corr linear scorer is neutral/negative (`0.6136`), but temporal clip-global assignment improves to `0.6831`; simple agreement policy reaches `0.6881`; oracle window/global selector reaches `0.8914`, proving large headroom if the temporal identity selector is solved.
- 2026-07-07: E11-A7/A8/A9/A10 completed through tmux. A7 train-supervised orientation (`0.5469`) and A8 Viterbi (`0.5960`) are negative; A9 signed lag-correlation improves window matching to `0.6586`; A10 hybrid signed-window / abs-global consistency policy reaches `0.9014`, exceeding E9 raw-keypoint direct seed0 `0.8396`. Caveat: A10 threshold was selected on the current 7-clip sweep.
- 2026-07-07: E11-A11 completed through tmux. Frozen A10 rule validated against E9 direct custom seeds 0/42/123: A10 `0.9014` vs `0.8396 / 0.6364 / 0.4279`; delta vs 3-seed per-clip mean `+0.2667` with bootstrap CI `[+0.1516, +0.3680]`. Delta vs strongest seed0 is positive but CI crosses zero.
- 2026-07-07: E12-A1 completed through tmux. Training the same E11 selector family on EgoHumans realistic source across folds `0/1/2/3/42/123` always selects `abs_global`; frozen custom result is `0.6831 ± 0.0000`. This beats MoBind-like fine-tune `0.5653`, E9 corrected transfer `0.6148`, and E9 direct custom 3-seed mean `0.6346`, but does not reproduce custom-selected A10 `0.9014`.
- 2026-07-07: E13-A1 opened the physics-strengthened rule distillation route. Research notes cover rule distillation, Soft-DTW/differentiable temporal alignment, differentiable assignment/ranking, and a raw-pose + kinematic auxiliary student model.
- 2026-07-07: E13-A2/A3 completed through tmux. A2 generated seven clip-level teacher-signal files. A3 trained a leave-one-clip-out model+rule selector over signed-window vs abs-global decisions; best result is `0.6998`, slightly above E12 `0.6831` but below G4/E11 dual fusion `0.7516`. Conclusion: clip-level confidence features are not enough; continue with pair/window-level teacher distillation.
- 2026-07-07: E13-A4/A4b completed through tmux. MLP pair scorer over lag-correlation features reaches mean train accuracy `0.9888` but eval FrameAcc only `0.5989 +/- 0.0127` over seeds `0/1/2/3/42/123`, below E12/A3. Conclusion: pair-only student overfits and does not solve temporal identity assignment.
- 2026-07-07: E13-A5 completed through tmux. Window-level temporal selector chooses between signed-window and abs-global assignments. After removing a GT-derived prototype feature and regenerating no-leak outputs, clip-CV reaches `0.9014`; stricter session-CV reaches `0.8251`, exceeding G4/E11 dual fusion mean `0.7516`. Main failure case remains `custom_20260211_171423_seg1` under session holdout.
- 2026-07-07: E13-A6 completed through tmux on GPU7. Neural MLP window selector trained on the same A5 no-leak features is negative over seeds `0/1/2/3/42/123`: clip-CV `0.5908 +/- 0.0450`, session-CV `0.5898 +/- 0.0068`. Conclusion: the current small neural selector overfits/calibrates poorly; A5 ridge remains the positive model+rule result.
- 2026-07-07: E14 opened for shoulder-local arm-vector Spatial->Temporal modeling. A1/A1b show the best stable vector variant reduces jitter (`0.4127x` raw z-jitter) but loses IMU lag-corr. A2 neural spatiotemporal pair model overfits train pairs (`0.9955`) and evals at only `0.3510`.
- 2026-07-07: E14-A3 session-split overfit check completed through tmux for seeds 0/42/123. Four-fold protocol uses one session as test, one as val, and two as train. Held-out session FrameAcc is `0.5026 +/- 0.1250`, so session-level generalization remains unstable.
- 2026-07-07: E14-A4 augmented original-split pair model completed through tmux/GPU for seeds 0/42/123. It reaches `0.9005 +/- 0.0119`, but A4 leakage audit shows the original split is not independent: `309/455` eval sliding windows overlap train cache windows and all seven train/test `(session, segment)` keys intersect. Treat A4 as a leaky regularization sanity check; current non-leaky result remains A3 session-heldout `0.5026 +/- 0.1250`.
- 2026-07-07: E14-A5 augmented session-heldout completed through tmux/GPU for seeds 0/42/123. Applying A4 augmentation under the non-leaky one-session-test protocol yields `0.4359 +/- 0.1072`, below A3 non-augmented `0.5026 +/- 0.1250`. Conclusion: masking/noise/rotation does not solve cross-session generalization for the pair scorer.
- 2026-07-07: E14-A6 Protocol A feasibility audit completed. Holding out the seven evaluation clips leaves `0` non-evaluation supervised extracted/cache clip candidates in the current E9/E4b/G3 custom data, so Protocol A cannot be run without a separate data-integration step.
- 2026-07-07: E15 data integration completed. A1 full-session tails produced `0` usable windows because annotation coverage ends at the eval clips. A2 converted external supervised custom NPZ data to E9-compatible `pose2d + LeftWrist.npy` cache: 4589 windows, train 3863 / val 726. Caveat: source placement is `L_LowArm` projected to 7D, not true LeftWrist.
- 2026-07-07: E16 MoBind best-config one-session-out 4fold launched through detached tmux. Folds use one session as test, one as val, and two as train.
- 2026-07-07: E17 opened as the next model-design route after E14. Diagnosis: E14's token mean pooling destroys arm topology. New design uses shoulder-rooted hierarchical typed arm graph tokens (bone-level, part-level, global-context), explicit relation features, topology-aware spatial encoder, typed physical IMU tokens, temporal CNN + GRU/Transformer ablation, and contrastive retrieval training under one-session-out validation.
- 2026-07-07: E16 completed. MoBind direct custom LeftWrist one-session-out seed0 4fold is `0.5998 +/- 0.2297` with folds `0.4125 / 0.6753 / 0.3694 / 0.9421`.
- 2026-07-07: E17-A1/A2/A3 completed. Architecture preflight passed, but the first topology-aware contrastive model is negative under one-session-out: `0.3789 +/- 0.1760` over four folds. Do not expand this exact full model to multi-seed; next step, if any, should be ablation against raw 7D IMU tower, E14-style pooling, and bone-only tokens.
- 2026-07-07: E17-A4 targeted ablations completed through tmux. `raw_imu` is best at `0.5087 +/- 0.1215`; `mean_pool=0.4354 +/- 0.0279`; `bone_only=0.3924 +/- 0.1918`; `no_temporal_cnn=0.3762 +/- 0.0846`. Main diagnosis: typed IMU physical tokens are the largest overfitting source; skeleton topology attention also overfits; temporal CNN is not the primary culprit.
- 2026-07-07: E18 completed through tmux. Previously trained source-only EgoHumans realistic LeftWrist checkpoints were audited to ensure they had not seen custom data, then evaluated on E16 one-session-out folds. Result: `0.5009 +/- 0.2476` over 12 source-seed/fold evaluations, weighted FrameAcc `0.5043`, below E16 direct custom `0.5998 +/- 0.2297`.
- 2026-07-07: E19 anti-shortcut protocol source patch completed. Added shared protocol module, patched E17 to audit session/window split by default, added label-shuffle / temporal-shift / static temporal controls, and verified the path with tmux smoke runs. Future G5 model claims must report aligned result plus sanity controls.
- 2026-07-07: E20 completed the requested non-leaky multi-seed final comparison across direct custom MoBind, MoBind realistic transfer, and E17 realistic transfer. All three methods have 12/12 results and A5 summary; realistic transfer does not beat direct custom under held-out-session testing.
- 2026-07-08: Opened E21 and E22 planning nodes. E21 tests large-scale shoulder-local vector pretraining before custom transfer; E22 tests raw IMU plus spectral/audio-style auxiliary features.
- 2026-07-08: E21-A0/A1 completed through tmux. Realistic LeftWrist source cache is usable (train 3475 / val 228 / test 956). Source-domain sanity: raw_pose time `0.6590`, vector time `0.6967`, hybrid time `0.7427` versus chance `0.3439`; spectral IMU summaries reduce the best setting.
- 2026-07-08: E21 follow-up constraint added: do not rerun E17 raw-IMU realistic transfer, because E20 already completed that path over 4 folds x 3 seeds and found `0.4685 ± 0.2287`. E21-A2 must be a true raw-keypoint + shoulder-vector hybrid source model.
- 2026-07-08: E21-A2/A3 completed through tmux/GPU. True hybrid raw-pose + shoulder-vector model trained on realistic source and transferred to E16 one-session-out folds. Result: `0.6261 ± 0.1878` over 12 fold/seed runs; paired delta vs E20 MoBind direct custom is `+0.1313`, bootstrap CI `[+0.0135, +0.2451]`.
- 2026-07-08: E21-A4/A5 controls completed. Source zero-shot is lower (`0.4131 ± 0.1618`), so target fine-tuning is meaningful. Seed0 label-shuffle target fine-tune remains high (`0.5631 ± 0.0879`); aligned seed0 beats it by `+0.1663`, but CI crosses zero, so publication-level claims should extend controls to all seeds.
- 2026-07-08: E21-A6 same-architecture direct custom/no-source-pretrain ablation completed through tmux/GPU over 4 folds x 3 seeds. Result: `0.5120 ± 0.1987`. Source-pretrain fine-tune beats it by `+0.1141`, bootstrap CI `[-0.0027, +0.2524]`; hybrid representation alone is not enough to explain the E21 gain.
- 2026-07-08: E21-A7 extra seed robustness completed through tmux/GPU. New seeds `1/2/3` produce `0.6840 ± 0.1718`; all six seeds `0/1/2/3/42/123` produce `0.6550 ± 0.1823` over 24 fold-runs. Result/file audit found no immediate bug; fold2 is unusually easy/high and should be treated as a caveat.
- 2026-07-08: E21-A8 full label-shuffle control completed through tmux/GPU for all 6 seeds x 4 folds. Label-shuffle is `0.4628 ± 0.1429`, aligned is `0.6550 ± 0.1823`, paired delta `+0.1923` with bootstrap CI `[+0.1123, +0.2769]`. This resolves the earlier seed0-only inconclusive control in favor of E21, while keeping fold1/fold2 caveats.
- 2026-07-08: E22-A1 completed through tmux over E16 one-session-out folds. Time-domain diagnostic weighted `0.6709`, spectral-only `0.5316`, time+spectral `0.5443`; do not launch heavy spectral neural training yet.

## 关键结果

| 设置 | Zero-shot (none) | Fine-tune fusion (none) | 备注 |
|---|---|---|---|
| G5/E1 EgoHumans source (seed0) | 0.2940 | 0.7332 | 域差距大 |
| G5/E2 Realistic source (6 seeds) | 0.5283 ± 0.0740 | 0.6273 ± 0.1228 | 方差高，seed123 异常低 |
| E4b MoBind-like single-source | 0.4495 (seed0) | 0.5653 ± 0.0113 (3 seeds) | 弱迁移源 |
| E4b Realistic single-source | 0.5835 ± 0.1300 (4 source seeds; seed0=0.7077) | 0.6928 ± 0.0604 (3 seeds) | 强于 MoBind-like，但 zero-shot 方差大 |
| G5/E3 Realistic single-source low-LR | — | 0.7413 ± 0.0241 (6 seeds) | Historical mislabeled-wrist baseline; superseded by E9 corrected LeftWrist result |
| G5/E9 Realistic LeftWrist single-source | 0.6115 ± 0.1409 (3 seeds) | 0.6148 ± 0.1638 (3 seeds) | Corrected LeftWrist transfer; does not beat direct custom on matched seeds |
| G5/E9 Direct custom LeftWrist | — | 0.6346 ± 0.1681 (3 seeds) | Corrected LeftWrist direct custom baseline; high seed variance |
| G5/E11 A1 traditional kinematic matcher | — | 0.6352 (seed0/no training) | Shoulder-anchored kinematic signal is real and beats learned low-dimensional replacement |
| G5/E11 A3 50D kinematic MoBind feature path | — | 0.3355 (seed0) | Negative result; existing feature interface is inappropriate |
| G5/E11 A5 temporal kinematic assignment | — | 0.6831 (seed0/no training) | Best deployable kinematic matcher so far; improves over A1 and E9 corrected 3-seed direct mean |
| G5/E11 A6 oracle selector | — | 0.8914 (oracle) | Not deployable; proves temporal selector headroom |
| G5/E11 A10 hybrid consistency selector | — | 0.9014 (current split) | First differential/kinematic result above raw-keypoint seed0; threshold selected on current sweep |
| G5/E11 A11 frozen-rule multi-seed validation | — | +0.2667 vs direct 3-seed per-clip mean | Supports differential route against available multi-seed direct baselines |
| G5/E12 source-trained kinematic selector | — | 0.6831 ± 0.0000 (6 source folds) | Realistic source training selects abs-global; beats MoBind-like/E9 corrected means but not custom-selected A10 |
| G5/E13 A3 LOOCV model+rule selector | — | 0.6998 | Weak positive over E12 abs-global; not significant enough |
| G5/E13 A4 MLP pair scorer | — | 0.5989 +/- 0.0127 | Negative across 6 seeds; high train accuracy but poor eval |
| G5/E13 A5 window temporal selector | — | 0.8251 session-CV / 0.9014 clip-CV | First positive model+rule result; held-out session protocol exceeds G4/E11 dual fusion mean |
| G5/E13 A6 MLP window temporal selector | — | 0.5898 session-CV / 0.5908 clip-CV | Negative neural selector; trained on GPU over 6 seeds |
| G5/E14 A3 session-split vector pair model | — | 0.5026 +/- 0.1250 | One session test / one val / two train; unstable held-out session generalization |
| G5/E14 A4 augmented original-split vector pair model | — | 0.9005 +/- 0.0119 | Leaky sanity check: 309/455 eval windows overlap train windows; not valid generalization |
| G5/E14 A5 augmented session-split vector pair model | — | 0.4359 +/- 0.1072 | Non-leaky augmentation result; worse than A3 |
| G5/E16 MoBind direct custom LeftWrist one-session-out | — | 0.5998 +/- 0.2297 | Best existing MoBind config under non-leaky 4-session protocol |
| G5/E17 topology-aware arm-vector model | — | 0.3789 +/- 0.1760 | Negative seed0 4fold; full model overfits quickly and underperforms E16/E14-A3 |
| G5/E17 raw-IMU ablation | — | 0.5087 +/- 0.1215 | Best E17 ablation; points to typed IMU tower as main failure source |
| G5/E18 realistic source-only zero-shot one-session-out | 0.5009 +/- 0.2476 | — | Uses E9 A3 EgoHumans-only checkpoints; below E16 direct custom |
| G5/E20 MoBind direct custom one-session-out | — | 0.4947 +/- 0.2326 | 4 folds x 3 seeds; final non-leaky direct baseline |
| G5/E20 MoBind realistic transfer one-session-out | — | 0.4504 +/- 0.1780 | 4 folds x 3 seeds; below direct custom |
| G5/E20 E17 realistic transfer one-session-out | — | 0.4685 +/- 0.2287 | 4 folds x 3 seeds; below direct custom |
| G5/E21 hybrid vector direct custom one-session-out | — | 0.5120 +/- 0.1987 | Same architecture as E21, no source pretrain; hybrid expression alone gives modest gain |
| G5/E21 hybrid vector realistic transfer one-session-out | 0.4131 +/- 0.1618 | 0.6550 +/- 0.1823 | 4 folds x 6 seeds; label-shuffle control 0.4628 +/- 0.1429; best learned transfer setting |
| G5/E4 Realistic dual low-LR seed123 | — | 0.4861/0.4894 (single diagnostic seed) | 修复 E2 seed123 但远低于 E3，不扩展 |
| G4/E11 custom from-scratch | — | 0.752 ± 0.095 | 当前 SOTA |

- **最佳单点**：E2 seed0 fine-tuned fusion with `minmax` normalization = **0.8294**。
- **历史最佳 transfer 多 seed**：E3 realistic single-source low-LR fine-tune = **0.7413 ± 0.0241**，但该结果使用 RightWrist motion target。E9 corrected LeftWrist transfer 完成后为 **0.6148 ± 0.1638**，不再支持“旧 E3 是有效左腕结论”的解释。
- **异常 seed**：E2 seed123 在 zero-shot（0.3787）与 fine-tune（0.4048）中均显著偏低；E4 low-LR dual 只能修到 0.4894，说明问题不只是学习率。
- **主要发现**：realistic source 大幅降低 zero-shot 域差距；保守 target fine-tune 是当前最有效的跨数据集提升；dual-embedding 迁移在当前设置下仍不稳定。

## 待办

1. E17 full topology-aware contrastive model 已验证为负结果；A4 ablation 定位主要问题在 typed IMU physical-token tower，其次是 skeleton topology attention。若继续 E17，应改成 raw 7D IMU tower + 低容量 skeleton encoder，而不是继续 full attention/typed IMU。
2. E13-A5 证明 temporal/window-level selector 是有效方向：session-CV `0.8251` 已超过 G4/E11 dual fusion mean。下一步应冻结 A5 no-leak feature set，分析 `171423_seg1` 失败原因，并考虑 source/pretrain 或更稳的 regularized selector。
3. E15 已补出一个可训练的外部 custom cache；下一步若继续 Protocol A，应先决定是否接受 `L_LowArm -> LeftWrist-format` placement mismatch，然后用它做 train/val、七个 eval clips 做 test。
4. E16 MoBind one-session-out `0.5998 +/- 0.2297` 是当前 non-leaky MoBind baseline。
5. E18 source-only realistic pretrain one-session-out `0.5009 +/- 0.2476` 低于 E16，说明没有 custom fine-tune/训练时，source-only checkpoint 不足以替代 custom session-out training。
6. E19 后，未来 G5 model result 必须同时报告 aligned session-out 与 label-shuffle / temporal-shift / static controls；普通 `DataLoader shuffle=True` 不再视为 anti-shortcut 证据。
7. E20 多 seed 最终对比确认：在严格 one-session-out 下，MoBind/E17 的 realistic source transfer 都没有超过 direct custom；旧同源 clip/window 结果不能作为跨 session 泛化证据。
8. E21 已完成 4 folds x 6 seeds positive transfer result，并补齐 same-architecture direct-custom ablation 与 full label-shuffle control。结论是 hybrid 表达本身只有小幅收益，realistic source pretrain + custom fine-tune 组合贡献最大；aligned-shuffle paired delta CI 为正。下一步若继续，应分析 fold1 弱、fold2 过高/过易的原因，或补 temporal-shift/static controls。E22 spectral/audio-style feature 审计为负，暂不启动 heavy spectral neural branch。
9. 暂停 E4 realistic dual transfer 扩展，除非先定位 global branch / fusion failure。
10. 对 transfer 主线，target-only light jitter 已验证无收益；下一步若继续 E6，应做 source+target 同步 canonicalization 或更明确的 acc frame/gravity 处理。
11. adaptive gate 可作为 direct custom dual/deployment 稳定性方向，但不作为拯救 E2 transfer dual 的主线。
12. 暂缓 TotalCapture multi-source，直到实现单臂 sensor mapping 或 placement adapter。
