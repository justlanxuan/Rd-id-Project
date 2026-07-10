# G5 Plan: Cross-Dataset Transfer and Single-IMU Robustness Roadmap

## Current Diagnosis

G5 的核心问题已经从“能否做 EgoHumans -> custom 迁移”推进为：

> 如何为 **左手腕单 IMU** 场景学习一个跨数据集稳定的 IMU-video identity representation，使模型不依赖某一个数据集、某一个 source seed 或某一种 synthetic IMU convention？

**2026-07-06 correction:** custom IMU was worn on LeftWrist, not RightWrist.
Therefore E3/E4/E5/E6 numbers that used RightWrist motion targets are retained
as historical mislabeled-wrist baselines. The current required validation is
E9: explicit LeftWrist source pretraining -> custom zero-shot / fine-tune, plus
direct custom LeftWrist training.

当前已知事实：

| Evidence | Result | Interpretation |
|---|---:|---|
| G5/E1 EgoHumans source dual-embedding zero-shot | 0.2940 | MoBind-like EgoHumans -> custom domain gap 很大 |
| E4b MoBind-like single-source transfer | 0.4495 / 0.5653 ± 0.0113 | MoBind-like synthetic 不是好的 custom 迁移源 |
| E4b realistic single-source transfer | 0.5835 ± 0.1300 / 0.6928 ± 0.0604 | realistic synthetic 明显更接近 custom，但 zero-shot 方差大 |
| G5/E2 realistic dual-embedding fine-tune | 0.6273 ± 0.1228 (none), 0.6517 ± 0.1289 (minmax) | dual-embedding 未稳定超过 E4b single-source |
| G5/E3 realistic single-source low-LR fine-tune | 0.7413 ± 0.0241 | Historical mislabeled-wrist baseline; superseded by E9 corrected LeftWrist result |
| G5/E9 realistic LeftWrist single-source | 0.6115 ± 0.1409 / 0.6148 ± 0.1638 | Corrected LeftWrist zero-shot / low-LR fine-tune; does not beat direct custom |
| G5/E9 direct custom LeftWrist | 0.6346 ± 0.1681 | Corrected direct custom baseline; high seed variance |
| G5/E11 A1 traditional kinematic matcher | 0.6352 | No-training shoulder-anchored matcher; confirms kinematic signal is usable |
| G5/E11 A3 50D kinematic feature path | 0.3355 | Negative result; MoBind feature-only interface is the wrong model path |
| G5/E11 A5 temporal kinematic assignment | 0.6831 | Best deployable kinematic matcher so far |
| G5/E11 A6 oracle selector | 0.8914 | Not deployable; shows temporal selector headroom exceeds raw-keypoint seed0 |
| G5/E11 A10 hybrid consistency selector | 0.9014 | Current-split proof-of-concept above raw-keypoint seed0; threshold selected from sweep |
| G5/E12 source-trained kinematic selector | 0.6831 ± 0.0000 | Realistic source 6-fold training selects abs-global; deployable but below A10 |
| G5/E13 A3 LOOCV model+rule selector | 0.6998 | Weak positive over E12 abs-global; not enough for target |
| G5/E13 A4 MLP pair scorer | 0.5989 +/- 0.0127 | Negative across 6 seeds; overfits train pairs |
| G5/E13 A5 window temporal selector | 0.8251 session-CV / 0.9014 clip-CV | First held-out model+rule result above G4/E11 dual fusion mean |
| G5/E13 A6 MLP window temporal selector | 0.5898 session-CV / 0.5908 clip-CV | Negative neural selector over 6 seeds |
| G5/E14 A2 shoulder-local vector spatiotemporal pair model | 0.3510 seed0 | Negative; train pair acc 0.9955 but eval collapses |
| G5/E4 realistic dual low-LR seed123 diagnostic | 0.4861/0.4894 | 只能部分修复 E2 seed123，不值得扩到 6 seeds |
| G4/E11 direct custom dual fusion | 0.7516 ± 0.0946 | 当前 custom SOTA，但仍有 seed 方差 |

**结论：** 下一阶段不应盲目继续扩大源域训练。当前最有效的跨数据集能力来自 realistic source + conservative target adaptation；下一步应优先解决三件事：

1. single left-wrist IMU 的跨域物理表示一致性；
2. source pretrain -> custom fine-tune 的稳定性，默认以 E9 LeftWrist 结果作为最终 transfer baseline；
3. local/global dual-embedding 的自适应融合，而不是固定或按 seed 选 α。

---

## Phase 0: Documentation and Reproducibility Lock

### E0:transfer_result_audit

- **目标：** 把 E1 / E2 / E4b / G4/E11 的 cross-dataset 对照固定成一个可复查的结果锚点。
- **动作：**
  - 汇总所有 raw JSON 到统一表格，明确 source seed、target seed、window、branch、fusion norm。
  - 标注哪些结果是 single-run，哪些是 multi-seed mean ± std。
  - 明确当前结论：realistic single-source 是强迁移源；dual-embedding 当前均值不稳；direct custom dual fusion 仍是 SOTA。
- **产出：**
  - `results/cross_dataset_comparison.md`
  - `results/cross_dataset_comparison.json`

---

## Phase 1: Conservative Target Adaptation

### E3:realistic_single_source_conservative_finetune

- **科学问题：** realistic single-source 已经是最强 source；能否通过更保守的 fine-tune 稳定超过 0.6928 ± 0.0604，并接近 G4/E11 0.7516？
- **核心策略：** 不从 full fine-tune 开始，而是逐层控制可训练参数。

| 子实验 | 训练策略 | 目的 |
|---|---|---|
| A1 | freeze IMU encoder + motion encoder，只训练 projection / classifier head | 防止破坏 source 表示 |
| A2 | freeze Stage1 encoder，只 fine-tune Stage2 heads | 保留 contrastive 表示，适配 custom |
| A3 | progressive unfreezing: head -> Stage2 -> encoder | 控制过拟合 |
| A4 | low-LR full fine-tune | 判断 full fine-tune 是否只是学习率过大 |
| A5 | validation-selected checkpoint / early stop sensitivity | 检查 custom val 是否能预测 test |

- **基线：**
  - E4b realistic single-source fine-tune: 0.6928 ± 0.0604
  - G3/E2 direct custom w100 single local: 0.6588 ± 0.1304
  - G4/E11 direct custom w24 dual fusion: 0.7516 ± 0.0946
- **成功标准：**
  - 6 seeds mean ≥ 0.72 且 std ≤ 0.08，或
  - 不低于 direct custom dual fusion 的 lower range，同时显著低于其方差。

- **状态：** 已完成 A4 low-LR full fine-tune。6 seeds = `0.7413 ± 0.0241`，但该结果使用 RightWrist motion target；在 custom 左腕佩戴事实确认后，只作为 historical mislabeled-wrist baseline。E9 corrected LeftWrist revalidation 已完成，transfer mean 降为 `0.6148 ± 0.1638`。

### E4:realistic_dual_conservative_finetune

- **科学问题：** E3 的 low-LR 策略能否修复 G5/E2 dual local/global transfer 的高方差，尤其是 seed123 collapse？
- **状态：** 已完成 seed123 诊断。

| Norm | E2 seed123 | E4 low-LR seed123 | Delta |
|---|---:|---:|---:|
| none | 0.4048 | 0.4861 | +0.0813 |
| zscore | 0.4210 | 0.4894 | +0.0684 |
| minmax | 0.4234 | 0.4894 | +0.0660 |

- **结论：** low-LR 对 dual transfer 有帮助，但远不如 E3 single-source low-LR seed123 `0.7603`；暂不扩到 6 seeds。

---

## Phase 2: Adaptive Local/Global Fusion for Single-Wrist IMU

### E5:adaptive_local_global_gate

- **科学问题：** local branch 和 global branch 在不同 motion regime 下可靠性不同。能否用窗口级信号质量自适应选择 α，减少 seed-specific α tuning？
- **核心思路：**
  - local branch 适合佩戴腕运动丰富的窗口；
  - global branch 适合佩戴腕静止、遮挡、局部噪声大但全身动作仍有节律的窗口。

| 子实验 | Gate signal | 说明 |
|---|---|---|
| A1 | IMU acceleration / gyro energy | 只依赖 IMU，部署友好 |
| A2 | LeftWrist 2D velocity / visibility | 使用 skeleton 质量判断 local 是否可靠 |
| A3 | score entropy / margin | 两个模型自身置信度 |
| A4 | rule-based soft gate | 无额外训练，先验证可行性 |
| A5 | small learned gate on validation split | 轻量 MLP 或 logistic regression |

- **适用模型：**
  - G4/E11 direct custom dual models；
  - G5/E2 realistic-source dual models；
  - E4 conservative fine-tuned local/global models if dual branch failure is later fixed。
- **成功标准：**
  - 超过 G4/E11 fixed best-α 0.7516 ± 0.0946，或
  - 在不提高均值时显著降低 seed std。
- **状态：** A1 oracle diagnostics 已完成。G4 direct w24 frame oracle `0.8494 ± 0.1009`，说明 deployable gate 有上限空间；但 G5/E2 fine-tuned minmax clip oracle 仅 `0.6877 ± 0.1417`，低于 E3 low-LR transfer。因此 E5 不应作为拯救当前 transfer dual 的主线。

---

## Phase 3: IMU Canonicalization and Domain Randomization

### E6:imu_canonicalization_and_randomization

- **科学问题：** TotalCapture / EgoHumans / custom / realistic synthetic 的 IMU 坐标系、重力处理、传感器顺序和噪声统计不完全一致。能否通过统一物理表示与训练期扰动提升跨数据集能力？

### A1: IMU format audit

为每个数据源建立 machine-readable metadata：

| Field | Required |
|---|---|
| sensor_order | yes |
| channels | yes |
| coordinate_frame | yes |
| gravity_included | yes |
| sampling_rate | yes |
| unit_scale | yes |
| quaternion convention | yes |

### A2: Canonical input variants

比较不同输入表示：

1. `linear_acc_world + quat`（当前 MoBind-compatible 主线）
2. `acc_body + gyro + quat`
3. `acc_body + gyro + orientation-invariant features`
4. normalized acceleration magnitude + angular velocity magnitude

### A3: Domain randomization

训练期加入：

- random yaw / SO(3) rotation;
- scale jitter;
- bias drift;
- Gaussian noise;
- time shift / resampling jitter;
- sensor dropout / missing windows.

- **成功标准：**
  - zero-shot 和 fine-tune 方差下降；
  - source seed sensitivity 下降；
  - 不显著损害 direct custom performance。
- **状态：**
  - A1 IMU format audit 已完成。E4b/E3 的 20 Hz custom cache 文件名为 RightWrist，但 custom 实际佩戴 LeftWrist；E9 已重建 explicit LeftWrist source cache，并把 custom cache 语义改为 LeftWrist。TotalCapture 48D sensor placement 不匹配 left-wrist-only deployment，暂不宜直接混入。
  - A2/A3 target train-only light acc jitter 已完成 seeds `0/42/123`，结果与 E3 low-LR 完全持平，mean delta `-0.0000`。该增强强度不扩展。

---

## Phase 4: MoBInd Representation + Autism Pipeline Temporal Logic

### E7:mobind_embedding_pipeline_bridge

- **科学问题：** MoBInd 表示学习强，但 Autism pipeline 有更完整的数据处理、tracking、temporal voting 和部署评估逻辑。能否把 MoBInd embedding 接入 pipeline 后处理以提升跨数据集部署指标？
- **动作：**
  - 导出 MoBInd window-level IMU/video similarity matrix。
  - 接入 pipeline 的 sliding vote / temporal smoothing / identity consistency。
  - 用 FrameAcc、HOTA、AssA 同时评估。
- **成功标准：**
  - 在 custom 7 clips 上超过 raw MoBInd frame matching；
  - 对静态片段和 seed 异常片段有明确改善。

---

## Phase 5: Multi-Source Training with Dataset-Specific Adaptation

### E8:multi_source_dsbn_or_adapter

- **科学问题：** TotalCapture、EgoHumans、realistic synthetic 和 custom 是否可以共同训练一个 shared encoder，同时保留 dataset-specific normalization / adapter 以避免负迁移？
- **优先尝试顺序：**
  1. shared encoder + dataset-specific BN / AdaBN；
  2. shared encoder + lightweight dataset adapter；
  3. balanced source sampling；
  4. DANN only as secondary baseline, not primary method。
- **原因：** E4b 中 DANN 提升有限；IMU domain 差异可能是真实统计差异，不应强行完全 domain-invariant。

---

## Recommended Execution Order

| Priority | Experiment | Why now |
|---|---|---|
| P0 | E0 result audit | 已完成，锁定可复查基线 |
| P0 | E9 LeftWrist revalidation | 已完成；E3 只能作为 mislabeled historical baseline |
| P1 | E6 IMU canonicalization + randomization | 已完成格式审计与 target-only jitter；下一步必须 source+target 同步 canonicalization 或确认 acc 物理语义 |
| P2 | E5 adaptive gate | 对 direct custom/deployment 有空间，但不能救当前 E2 transfer dual |
| P2 | E7 MoBInd embedding -> pipeline temporal logic | 利用两套模型各自优势 |
| P3 | E8 multi-source DSBN / adapter | 在单源稳定后再做多源 |

## Immediate Next Step

E3 曾经回答核心问题，但现在只能作为历史基线：

> realistic source 已经比 MoBind-like 更接近 custom，且之前 transfer 主要受 target fine-tune 策略过激影响。

由于 custom 佩戴腕修正为 LeftWrist，E9 已完成 matched seeds 0/42/123：

1. realistic LeftWrist source -> custom zero-shot: `0.6115 ± 0.1409`；
2. realistic LeftWrist source -> custom low-LR fine-tune: `0.6148 ± 0.1638`；
3. direct custom LeftWrist from-scratch: `0.6346 ± 0.1681`；
4. corrected transfer 没有超过 direct custom，也不支持沿用旧 E3 结论。

E10 已开始诊断 motion-side representation。A1/A1b 的结论是：

1. filtered bone geometry / relative motion features 与 IMU dynamics 有信号；
2. smoothing improves alignment，支持 1D temporal CNN / low-pass front-end；
3. raw keypoints 仍有竞争力，因此下一步应做 hybrid representation，而不是纯差分替换；
4. A1b 补充验证了大臂-小臂夹角：elbow angle 是强几何特征，filtered elbow angular velocity 的 window-level energy correlation 最高到 `0.7050`。

E10-A3 已完成 direct custom seed0 小规模 A/B。低维 replacement-feature variants 全部失败：

1. `hybrid_v1`: `0.5330`；
2. `geometry_only`: `0.4445`；
3. `dynamics_only`: `0.4387`；
4. E9 raw-keypoint direct seed0 baseline: `0.8396`。

Immediate Next Step 改为停止低维替换方案；如果继续做差分路线，应构建 raw skeleton + auxiliary differential/angle channels，或双分支 raw-pose / differential-pose motion encoder。

E11-A1/A2/A3 对这个判断做了 sanity check：

1. no-training shoulder-anchored traditional matcher 达到 `0.6352`，超过 E10-A3 learned `hybrid_v1` replacement 的 `0.5330`；
2. 50D `shoulder_kinematic_v1` cache 保留了 LeftShoulder anchor、BodyCenter global context、upper/forearm geometry、elbow angle/velocity、wrist relative motion；
3. 但同一 50D state 通过 MoBind `motion_type: feature` direct seed0 只有 `0.3355`，远低于 traditional matcher 和 raw-keypoint direct seed0 `0.8396`。

E11-A4/A5/A6 进一步说明：

1. learned linear lag-correlation scorer `0.6136`，不如 A1；
2. clip-global temporal assignment `0.6831`，是当前最好的 deployable kinematic result；
3. simple agreement selector `0.6881`，只有小幅提升；
4. oracle window/global selector `0.8914`，超过 raw-keypoint seed0，但不可部署。

Immediate Next Step 改为解决 temporal identity selector。下一步如果继续 E11，应实现专用模型，而不是继续调 cache：

1. differentiable lag-correlation / cross-attention between IMU and kinematic channels；
2. 或 raw-pose encoder + kinematic auxiliary encoder 的 late fusion；
3. 或从 E11-A1 traditional matcher 的 window assignments 做 distillation / supervised pretext；
4. 或训练 A7 selector，在 window-nearest 与 clip-global / identity orientation 之间做 deployable selection。

E11-A10 now provides a current-split proof-of-concept that the
differential/relative-motion route can exceed raw keypoint matching:

- raw-keypoint direct seed0: `0.8396`
- A10 hybrid signed-window / abs-global consistency selector: `0.9014`

Important caveat: A10's U-shaped consistency threshold was selected from the
current 7-clip sweep. Treat it as evidence that the idea can work, not yet as a
final deployable method. The next rigorous step is to freeze the rule and run
robustness checks, or train the selector without test-sweep tuning.

E11-A11 completed that first robustness check against available direct-custom
multi-seed baselines:

- frozen A10 rule: `0.9014`
- direct custom seeds 0/42/123: `0.8396 / 0.6364 / 0.4279`
- direct custom 3-seed mean: `0.6346 +/- 0.1681`
- paired bootstrap vs 3-seed per-clip mean: `+0.2667`, CI
  `[+0.1516, +0.3680]`

This validates the HAROS objective at the current evidence level. Remaining
caveat: the frozen threshold was discovered on the same seven clips, so a future
publication-grade run should use new held-out recordings or train the selector
without test-sweep tuning.

E12 tested that source-trained version on EgoHumans realistic source. Across
source folds `0/1/2/3/42/123`, the selected frozen policy is always
`abs_global`, giving custom `0.6831 +/- 0.0000`. This is stronger than
MoBind-like fine-tune `0.5653`, E9 corrected transfer `0.6148`, and E9 direct
custom mean `0.6346`, but it does not reproduce the custom-selected A10
`0.9014`. The next step should therefore not be more threshold sweeps; it
should be a selector/model that can learn when custom needs signed-window
rather than abs-global without using the test clips.

E13 turns that into a model plan: use E11 rule outputs as teacher signals and
train a physics-augmented student. The planned student keeps raw skeleton
information, adds shoulder-anchored kinematic auxiliary channels, and uses
lag-aware/cross-attention plus assignment-aware losses. Immediate next step:
generate reusable teacher signals from signed-window, abs-global, A10-style
selector, score margins, and confidence.

E13-A2/A3 started that path. Clip-level teacher files were generated, and a
leave-one-clip-out model+rule selector reached `0.6998`. This is a weak positive
over E12 `0.6831`, but it is below G4/E11 dual fusion `0.7516` and far below
A10 `0.9014`. The next experiment must move from clip-level selector features
to pair/window-level teacher distillation.

E13-A4/A4b tested that pair-level direction with an MLP lag-correlation scorer.
It overfits train pairs (`0.9888` mean train accuracy) but reaches only
`0.5989 +/- 0.0127` FrameAcc over seeds `0/1/2/3/42/123`, so pair-only scoring
is not the missing piece. The next implementation
should optimize temporal assignment directly: soft Sinkhorn/Hungarian
distillation, sequence-level selector over window trajectories, or
cross-attention with temporal consistency loss.

E13-A5 validates that direction. A no-leak ridge window-level selector over
assignment trajectory features reaches `0.9014` with leave-one-clip-out and
`0.8251` with leave-one-session-out. This is the first model+rule result above
G4/E11 dual fusion mean `0.7516` under a held-out protocol. The next refinement
is not another pair scorer; it is stabilizing this temporal selector, especially
the session-held-out failure on `custom_20260211_171423_seg1`.

E13-A6 tested the user's requested neural training variant directly: a GPU
trained MLP on the same no-leak window features over seeds `0/1/2/3/42/123`.
It is negative (`0.5908 +/- 0.0450` clip-CV, `0.5898 +/- 0.0068` session-CV),
so the current evidence favors the simpler ridge model+rule selector over a
small neural selector on this dataset size.

E14 tests a more structured skeleton representation: shoulder-local / bone-vector
tokens with Spatial→Temporal neural encoding. The representation is promising as
a denoising auxiliary channel, but the first standalone neural pair model is
negative (`0.3510` seed0 despite `0.9955` train pair accuracy). Do not expand
this A2 design; if continuing, combine vector tokens with A5-style temporal
assignment supervision instead of pair-only BCE.

E14-A3/A4/A5 then clarified the failure mode. The original-split augmented
model can reach `0.9005 +/- 0.0119`, but that split is leaky: `309/455` eval
windows overlap train windows. Under the non-leaky one-session-out protocol,
the same vector-pair direction is only `0.5026 +/- 0.1250` without augmentation
and `0.4359 +/- 0.1072` with augmentation. The issue is not just regularization.
The architecture itself is too weak because it mean-pools `upper / forearm /
wrist_rel / context` tokens and destroys the topology needed to learn elbow
angle, forearm-vs-upper-arm relation, closure consistency, and context-arm
coupling.

E17 is therefore opened as the next model-design route, without overwriting
E14. E17 should use:

1. shoulder-rooted hierarchical typed arm graph tokens rather than unordered
   token bags: bone-level tokens (`upper_bone`, `forearm_bone`), part-level
   token (`arm_part`), and global token (`global_context`);
2. bone token features beyond `(dx, dy)`: direction, normalized/log length,
   smoothed velocity, acceleration, relative rotation where applicable, and
   visibility/confidence if available;
3. explicit relation features such as dot/cross, elbow angle, length ratio, and
   `wrist_rel - (upper + forearm)` closure residual;
4. a topology-aware per-frame spatial encoder with token type embeddings and
   relation-aware attention or relation MLP;
5. typed physical IMU tokens rather than only `Linear(7)->GRU`: `acc`,
   `acc_dyn`, `ori`, and `rot_dyn`, with both raw vector channels and invariant
   dynamics such as acceleration magnitude, acceleration delta magnitude, and
   quaternion angular speed;
6. temporal CNN front-end for both skeleton and IMU towers, then GRU vs
   Transformer ablation;
7. contrastive retrieval / InfoNCE with in-batch negatives instead of E14's
   binary pair classifier as the primary objective;
8. one-session-out validation, using E14-style token mean pooling, bone-only
   tokens, and raw 7D IMU
   tower only as ablation/control baselines.

E16 is currently running the best existing MoBind configuration under the same
one-session-out 4fold protocol. Once E16 finishes, its result becomes the
non-leaky MoBind baseline for E17.

E16 and the first E17 baseline have now completed. E16 MoBind direct custom
LeftWrist one-session-out seed0 is `0.5998 +/- 0.2297`. E17-A2, the first full
topology-aware contrastive model, is negative: `0.3789 +/- 0.1760`. It trains
and evaluates, but overfits almost immediately (`best_epoch = 1/18/1/2`) and is
below E16, E14-A3, and E14-A5. Do not expand this exact model to multiple seeds.
If continuing E17, use the result diagnostically and run targeted ablations:
raw 7D IMU tower, E14-style pooling, bone-only tokens, and simpler temporal
encoder variants.

Those targeted ablations are now complete. The best variant is `raw_imu`
(`0.5087 +/- 0.1215`), while `mean_pool` is more stable than full attention
(`0.4354 +/- 0.0279`), `bone_only` remains unstable (`0.3924 +/- 0.1918`), and
removing temporal CNN does not help (`0.3762 +/- 0.0846`). The current diagnosis
is that the typed IMU physical-token tower is the largest overfitting source,
and topology attention is also too high-capacity for this data size. Temporal
CNN is not the main culprit.

E20 then reran the three requested lines under a strict non-leaky one-session-out
multi-seed protocol: direct custom MoBind, MoBind realistic transfer, and E17
realistic transfer. All three completed 4 folds x 3 seeds. The final means are:

| Method | Mean FrameAcc | Std |
|---|---:|---:|
| MoBind direct custom | 0.4947 | 0.2326 |
| MoBind realistic -> custom transfer | 0.4504 | 0.1780 |
| E17 realistic -> custom transfer | 0.4685 | 0.2287 |

This confirms that the older high same-clip/window results do not transfer to
strict held-out sessions. Future model claims must first pass one-session-out
and anti-shortcut controls.

## Current Branches: E21 and E22

### E21: Large-scale shoulder-local vector pretraining

E14/E17 should not be expanded unchanged. The new vector question is whether
shoulder-local arm graph representations need large-scale source pretraining
before custom transfer. E21-A0/A1 completed the first source-domain sanity check:

1. raw-keypoint source baseline: `0.6590`;
2. shoulder-local vector-only source features: `0.6967`;
3. raw-keypoint + vector hybrid source features: `0.7427`;
4. chance baseline: `0.3439`.

This supports a small source-pretraining smoke for the hybrid vector route.
The default IMU tower should be conservative (`raw 7D + light dynamics`) because
E17-A4 identified the typed IMU physical-token tower as the largest overfitting
source. Do not add spectral IMU targets to the first E21 training smoke:
`hybrid + time+spectral` dropped to `0.6987`.

E21-A2/A3 then implemented the true hybrid model and completed 4 folds x 3 seeds:

1. source zero-shot: `0.4131 +/- 0.1618`;
2. source pretrain -> custom fine-tune: `0.6261 +/- 0.1878`;
3. E20 MoBind direct custom baseline: `0.4947 +/- 0.2326`;
4. paired delta vs direct custom: `+0.1313`, bootstrap CI `[+0.0135, +0.2451]`.

This is the first learned source-transfer model in G5 that beats the strict
one-session-out direct custom baseline. A4 seed0 label-shuffle remains high
(`0.5631 +/- 0.0879`), so next controls should extend label-shuffle/static tests
to all seeds before a final publication-level claim.

### E22: IMU spectral / audio-style preprocessing

E22 captures the IMU-as-audio idea. The design should not replace raw IMU with
FFT magnitude only. Instead, it should test raw 7D plus spectral/time-frequency
auxiliary features:

1. time-domain dynamics: acceleration magnitude, jerk, quaternion angular speed;
2. frequency summaries: band power, dominant frequency, spectral centroid,
   spectral entropy, low/high energy ratios;
3. time-frequency maps: STFT or wavelet/scalogram auxiliary branch;
4. skeleton-side matched dynamics: elbow angular velocity, wrist-relative
   velocity, body-center velocity, and optional spectral summaries.

E22-A1 completed the first no-training feature audit over E16 folds:

1. time-domain dynamics weighted `0.6709`;
2. spectral-only weighted `0.5316`;
3. time+spectral hybrid weighted `0.5443`.

This is negative for the current spectral/audio-style preprocessing. Neural
training should not follow until a refined auxiliary/filterbank design beats raw
time-domain dynamics under one-session-out.

不要继续把 E3/E6 右腕 motion target 结论当作最终结论；不要直接混入 TotalCapture 48D，除非先实现单臂 sensor mapping 或 placement adapter。
