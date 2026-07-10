# 🔄 HAROS Task Resume Node

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G5:cross_dataset_transfer — **发现 custom 实际为 LeftWrist IMU；E9 LeftWrist 复验已完成 matched seeds 0/42/123**。
* **上一阶段收尾:** G4:mobind_single_imu_adaptation — E11 dual-embedding 完成，A8 统计检验完成，SOTA 已更新为 w24 Fusion best α 0.752 ± 0.095（Commit: `dd61608`）。
* **当前具体执行任务:**
  - ✅ G5 goal 已初始化（formulation / survey / ideas / plan / progress）。
  - ✅ E1 实验目录已创建（formulation / plan / progress / scripts / configs）。
  - ✅ A1: EgoHumans cache 已构建（`cache_action_1.2_0.5`，21,179 windows）。
  - ✅ A3: Source Model-G seeds 0/1 已完成；seeds 42/123 训练中。
  - ✅ A2: Source Model-L seed0 已完成；seed42 训练中。
  - ✅ A5: Seed0 zero-shot 完成：Mean FrameAcc = **0.2940**。
  - ✅ A6: Seed0 Local fine-tune 完成：Mean FrameAcc = **0.7259**。
  - ✅ A6: Seed0 Global fine-tune 完成。
  - ✅ A6: Seed0 Fusion (local+global fine-tuned) 完成：Mean FrameAcc = **0.7332**（best α=0.9，接近 from-scratch 0.752）。
  - ✅ E0: cross-dataset comparison 已生成，见 `experiments/G5:cross_dataset_transfer/results/cross_dataset_comparison.md/json`。
  - ✅ E3: realistic single-source low-LR target fine-tune 已完成 6 seeds，结果 **0.7413 ± 0.0241**。
  - ✅ E4: realistic dual low-LR seed123 诊断已完成，best **0.4894**，只部分修复 E2 seed123 collapse，暂不扩展。
  - ✅ E5: adaptive gate oracle diagnostics 已完成；G4 direct w24 frame oracle **0.8494 ± 0.1009**，但 G5/E2 fine-tuned minmax clip oracle 只有 **0.6877 ± 0.1417**，不能救当前 transfer dual。
  - ✅ E6: IMU format audit 已完成；E4b/E3 的 **20 Hz custom RightWrist 7D** target cache 实际应按 LeftWrist 语义理解，TotalCapture 48D sensor placement 不匹配左手腕单 IMU，不宜直接混入。
  - ✅ E6-A2/A3: target train-only light acc jitter 已完成 seeds 0/42/123，结果与 E3 low-LR 完全持平，mean delta **-0.0000**，不扩展该增强。
  - ✅ E9: explicit LeftWrist source/custom cache 已完成；source 4659 windows，custom 316 windows。
  - ✅ E9-A6: direct custom LeftWrist from-scratch 已完成 seeds 0/42/123，FrameAcc = **0.8396 / 0.6364 / 0.4279**，mean **0.6346 ± 0.1681**。
  - ✅ E9 transfer: corrected LeftWrist source pretrain -> custom zero-shot/fine-tune 已完成 seeds 0/42/123；zero-shot **0.6115 ± 0.1409**，low-LR fine-tune **0.6148 ± 0.1638**。
  - ✅ E9 conclusion: corrected LeftWrist transfer does not beat direct custom LeftWrist **0.6346 ± 0.1681** on matched seeds; old E3 **0.7413 ± 0.0241** remains historical mislabeled-wrist baseline only.
  - ✅ E10-A1: skeleton differential / relative motion alignment diagnostic 已完成；filter sweep 显示 top best-lag corr 从 kernel1 **0.3613** 提升到 kernel9 **0.4341**，支持 low-pass/1D CNN 前端，但不支持纯差分替代 keypoints。
  - ✅ E10-A1b: 补充大臂-小臂夹角及其变化速度；elbow angle under smooth9 corr **0.4276**，elbow angular velocity window energy corr **0.7050**。
  - ✅ E10-A3: direct custom seed0 low-dimensional replacement-feature A/B 已完成；`hybrid_v1=0.5330`、`geometry_only=0.4445`、`dynamics_only=0.4387`，均显著低于 E9 raw-keypoint direct seed0 **0.8396**。不扩展这些替换方案。
  - ✅ E11-A1: no-training shoulder-anchored traditional kinematic matcher 已完成，FrameAcc **0.6352**，超过 E10-A3 learned `hybrid_v1` **0.5330**。说明 kinematic signal 可转化为 matching，E10-A3 的失败主要是表示压缩/模型接口问题。
  - ✅ E11-A2/A3: built 50D `shoulder_kinematic_v1` cache and trained direct custom seed0 through MoBind `motion_type: feature`; result **0.3355**. This is a negative result: richer kinematic state still fails through the current feature interface.
  - ✅ E11-A4/A5/A6: learned lag-corr scorer **0.6136**; temporal clip-global assignment **0.6831**; simple agreement selector **0.6881**; oracle window/global selector **0.8914**. The oracle is not deployable, but it proves large headroom if temporal identity selection is solved.
  - ✅ E11-A7/A8/A9/A10: A7 train-supervised orientation **0.5469** and A8 Viterbi **0.5960** are negative; A9 signed window **0.6586** improves local matching; A10 hybrid signed-window / abs-global consistency selector reaches **0.9014**, exceeding raw-keypoint direct seed0 **0.8396** on the current split.
  - ✅ E11-A11: frozen A10 rule multi-seed/bootstrap validation completed. A10 **0.9014** beats direct custom seeds **0.8396 / 0.6364 / 0.4279** and direct 3-seed mean **0.6346 ± 0.1681**; bootstrap vs 3-seed per-clip mean is positive.

## 2. 最终结果与结论
### G4/E11 w24 结果（6 seeds, sim_norm=none）
| 方法 | Mean FrameAcc | Std |
|---|---|---|
| Model-L (local) | 0.673 | 0.183 |
| Model-G (global) | 0.664 | 0.090 |
| Fusion α=0.5 | 0.709 | 0.124 |
| **Fusion best α** | **0.752** | **0.095** |

### G4/E11 w100 结果（6 seeds, sim_norm=none）
| 方法 | Mean FrameAcc | Std |
|---|---|---|
| Model-L (local) | 0.659 | 0.130 |
| Model-G (global) | 0.641 | 0.196 |
| Fusion α=0.5 | 0.675 | 0.135 |
| **Fusion best α** | **0.723** | **0.124** |

### A8 统计显著性检验结论（90,330 有效帧）
- **Local 与 Global 错误显著正相关，而非独立**：
  - w24: χ² = 13,993.74, p ≈ 0；w100: χ² = 18,546.71, p ≈ 0。
- **实际互补帧比例低于独立随机期望**：
  - w24: L-only 14.3% vs 期望 22.8%（比值 0.63）；G-only 12.8% vs 期望 21.3%（比值 0.60）。
  - w100: L-only 13.4% vs 期望 23.4%（比值 0.57）；G-only 11.8% vs 期望 21.8%（比值 0.54）。
- **但 L-only / G-only 帧具有显著时序结构**：
  - w24: L-only 平均连续段 39.2 帧，G-only 30.1 帧（随机 baseline ~1.16 帧）。
  - w100: L-only 23.4 帧，G-only 35.8 帧（随机 baseline ~1.15 帧）。
- **Fusion 仍然可靠地 rescue 了大部分单一模型错误帧**：
  - w24: 27.1% 的帧通过融合从单一模型错误变为正确；w100: 25.2%。
  - 极少出现单一模型对但 Fusion 错的情况（~1.4–1.7%）。

**综合判断**：Fusion 的收益是真实的，但 Local 与 Global 的互补性被高估——它们的错误模式高度相关，导致“只有一个模型对”的帧比理想独立专家少约 40–45%。Fusion 的上限因此受限，未来需要关注如何处理两者共同失败的片段。

### G5 新目标
- **核心问题：** 如何为左手腕单 IMU 学习跨数据集稳定表示，同时避免 custom 小数据 fine-tune 破坏 source representation？
- **当前结论：** realistic source 明显优于 MoBind-like source；低学习率 target fine-tune 是当前最有效的 transfer 改进，但 E3 的 0.7413 ± 0.0241 现在只能作为 historical mislabeled-wrist baseline。
- **当前修正结论：** E9 realistic LeftWrist transfer 已完成，low-LR fine-tune **0.6148 ± 0.1638**，低于 direct custom LeftWrist **0.6346 ± 0.1681**；下一步应诊断 seed123 collapse / motion-side sensitivity。
- **新想法整理：** E10/E11/E12/E13 已把骨架差分/角度/相对运动/全局中心运动假设落成 HAROS。E11-A11 已验证 frozen A10 rule 在 available multi-seed direct baselines 上仍成立：**0.9014** vs direct seeds **0.8396 / 0.6364 / 0.4279**。E12 realistic source-trained selector 6 folds 稳定选择 `abs_global`，custom **0.6831 ± 0.0000**，超过 MoBind-like/E9 corrected means，但未复现 custom-selected A10。E13-A3 clip-level LOOCV selector **0.6998**，只是弱正结果；E13-A4/A4b pair-only MLP 多 seed **0.5989 ± 0.0127**，负结果确认；E13-A5 window temporal selector no-leak clip-CV **0.9014**、session-CV **0.8251**，是当前首个显著正结果；E13-A6 GPU-trained neural MLP window selector **0.5898 ± 0.0068** session-CV，负结果。
- **E14 新想法：** shoulder-local arm-vector Spatial→Temporal 表征已按 HAROS 推进。A1/A1b 显示稳定 vector 表征降噪但降低 IMU corr；A2 无增强 pair model seed0 eval **0.3510**。A3 session-heldout（1 session test / 1 val / 2 train）为 **0.5026 ± 0.1250**，仍不稳。A4 原划分增强版达到 **0.9005 ± 0.0119**，但泄露审计确认 original split 的 train/eval clip 共源且 **309/455** eval windows 与 train windows 重叠。A5 把同样增强放到非泄露 session-heldout 后只有 **0.4359 ± 0.1072**，低于 A3。A6 Protocol A 审计显示当前 converted custom data 没有七个 eval clips 之外的监督 train clips。
- **E15 data integration:** A1 full-session tail route produced **0** usable windows because annotation coverage ends at the eval clips. A2 converted external supervised custom NPZ data to E9-compatible cache: **4589 windows**（train **3863** / val **726**）with `pose2d (100,2,17)` and `LeftWrist.npy (100,7)`. Caveat: A2 uses `L_LowArm -> acc3+quat4`, not true LeftWrist placement.
- **E16 completed:** MoBind best-config one-session-out 4fold seed0 is **0.5998 ± 0.2297**.
- **E17 completed A1/A2/A3/A4:** topology-aware arm-vector model preflight passed, but the first full contrastive model is negative: **0.3789 ± 0.1760** over 4 folds. Targeted ablations are complete: `raw_imu=0.5087 ± 0.1215`, `mean_pool=0.4354 ± 0.0279`, `bone_only=0.3924 ± 0.1918`, `no_temporal_cnn=0.3762 ± 0.0846`. Diagnosis: typed IMU tower is the largest failure source; topology attention also overfits.
- **E18 completed:** source-only realistic EgoHumans LeftWrist pretrain zero-shot on custom one-session-out is **0.5009 ± 0.2476** over 3 source seeds x 4 folds; lower than E16 direct custom **0.5998 ± 0.2297**.
- **E19 completed source patch:** added shared anti-shortcut protocol helpers and patched E17 so future runs audit split leakage by default and can run label-shuffle, temporal-shift, static skeleton, static IMU, and time-reversal controls. E16 fold audit passes for all four folds.
- **E20 completed:** final non-leaky one-session-out multi-seed comparison finished with 4 folds x 3 seeds per method. MoBind direct custom **0.4947 ± 0.2326**, MoBind realistic transfer **0.4504 ± 0.1780**, E17 realistic transfer **0.4685 ± 0.2287**.
- **E21/E22 current status:** E21 has progressed from positive source sanity to a learned source-transfer positive result. Hybrid raw-pose + shoulder-vector source pretrain -> custom one-session-out fine-tune is now **0.6550 ± 0.1823** over 4 folds x 6 seeds after A7 expanded seeds `1/2/3`; the new seeds alone are **0.6840 ± 0.1718**. E21-A6 same-architecture direct custom/no source pretrain is **0.5120 ± 0.1987**. E21-A8 full label-shuffle control is **0.4628 ± 0.1429**, aligned-shuffle paired delta **+0.1923**, CI **[+0.1123, +0.2769]**. Interpretation: hybrid alone gives only modest gain; hybrid + realistic source pretrain + custom fine-tune is the strongest learned transfer combination so far. Result audit found no immediate file/config/split bug, but fold1 is weak and fold2 is unusually easy/high. E22 spectral diagnostic is negative (time-domain **0.6709**, spectral-only **0.5316**, hybrid **0.5443**), so do not launch a heavy spectral branch yet.
- **E21-A9 source-domain check:** trained hybrid checkpoints achieve **0.9829 ± 0.0065** FrameAcc on held-out realistic EgoHumans source test over seeds `0/1/2/3/42/123`. Leakage audit passed with zero train/test and val/test sequence, exact-key, and content-hash overlap. This means source learning is strong; the lower custom scores reflect domain transfer, not source underfitting.
- **E20-A6 running:** user requested three additional seeds for the non-hybrid MoBind realistic source-pretrain -> custom fine-tune baseline. Detached tmux sessions `e20_a6_mobind_extra_seed1/2/3` are running source seeds `1/2/3`; they are currently in E9 realistic LeftWrist source Stage1 pretraining. After completion, E20 A5 should summarize MoBind realistic transfer over seeds `0/1/2/3/42/123` (24 fold-runs) for direct comparison with E21 hybrid 6-seed result.

## 3. 当前阻塞痛点 (Blockers & Issues)
* 默认沙箱内仍可能看不到 GPU；GPU 命令和长训练需使用已批准的非沙箱执行，并且长任务全部通过 detached `tmux` 跑。
* 注意：`.log/resume.md` 与 `experiments/SOTA_reproduce.md` 已提交至 Git；G5/E1 文档在 `experiments/` 目录下，按 `.gitignore` 不进入 Git，但本次通过 `git add -f` 可显式提交 goal 级文档。

## 4. 下一步行动 (Next Actions)
* [x] G5 goal 文档已提交 Git（Commit: `d10055c`）。
* [x] E1-A1 EgoHumans cache 已构建完成。
* [🔄] E1-A2/A3 source 训练 seed0 验证中。
* [ ] 待 seed0 成功后，扩展至 6 seeds 并行训练。
* [x] 2026-07-05：完成 E4b realistic/MoBind-like single-source transfer 与 direct custom training 的结果核对，并补充相关文档对照表；已区分 realistic seed0 zero-shot 0.7077 与 4-source-seed zero-shot 0.5835±0.1300。
* [x] 2026-07-05：按用户要求更新 `experiments/G5:cross_dataset_transfer` 总体计划，下一阶段优先 E3 conservative fine-tune、E4 adaptive local/global gate、E5 IMU canonicalization/domain randomization。
* [x] 2026-07-05：完成 E0 comparison 生成脚本与结果；E3 low-LR fine-tune 6 seeds = 0.7413±0.0241；E4 realistic dual low-LR seed123 = 0.4861/0.4894，暂不扩展。
* [x] 2026-07-05：完成 E5 adaptive gate oracle 诊断；结论是 adaptive gate 对 direct custom/deployment 有空间，但 alpha selection alone 不能超过 E3 transfer。
* [x] 2026-07-05：完成 E6 IMU format audit；结论是继续走 E3/E4b 20 Hz 7D acc+quat path，暂缓 TotalCapture 48D multi-source。
* [x] 2026-07-05：完成 E6-A2/A3 target train-only light acc jitter；seeds 0/42/123 与 E3 low-LR 完全持平，不扩展。
* [x] 2026-07-07：E9 corrected LeftWrist matched seeds 0/42/123 已完成；A7 summary 已重生成。当前不扩展到 6 seeds，优先诊断 seed123 collapse / motion-side sensitivity。
* [x] 2026-07-07：E10-A1 通过 tmux 完成 skeleton-IMU alignment diagnostic 与 filter sweep。
* [x] 2026-07-07：E10-A3 通过 tmux 完成 replacement-feature direct custom seed0 A/B；低维 hand-crafted geometry/dynamics/hybrid 均显著低于 raw-keypoint baseline，不扩展。
* [x] 2026-07-07：E11-A1 通过 tmux 完成 shoulder-anchored traditional matching sanity check；结果 **0.6352**，支持继续设计专用 kinematic model。
* [x] 2026-07-07：E11-A2/A3 通过 tmux 完成 50D shoulder-anchored kinematic cache + MoBind feature-path direct seed0；结果 **0.3355**，确认应停止 feature-only replacement，转向专用 lag/cross-attention 或 raw-pose + kinematic auxiliary architecture。
* [x] 2026-07-07：E11-A4/A5/A6 通过 tmux 完成 learned lag-corr 与 temporal assignment；当前 deployable kinematic result **0.6831-0.6881**，oracle selector **0.8914**。目标尚未完成，下一步要做可部署 selector / cross-attention。
* [x] 2026-07-07：E11-A7/A8/A9/A10 通过 tmux 完成 selector diagnostics；A10 hybrid consistency selector **0.9014**，首次超过 raw-keypoint direct seed0 **0.8396**。下一步建议 freeze A10 rule 并做 robustness check。
* [x] 2026-07-07：E11-A11 通过 tmux 完成 frozen-rule multi-seed/bootstrap validation；支持差分/相对运动路线超过当前 raw-keypoint direct baseline family。
* [x] 2026-07-07：E12-A1 通过 tmux 完成 realistic source-trained selector 多 source-seed 验证；6 folds 全部选择 `abs_global`，custom **0.6831 ± 0.0000**。结论：source-trained kinematic matching 超过 MoBind-like/E9 corrected mean，但 A10 的 **0.9014** 仍只是 custom-selected proof-of-concept。
* [x] 2026-07-07：E13-A1 完成联网调研、brainstorming 和 HAROS 计划。下一步是 E13-A2 teacher-signal generation。
* [x] 2026-07-07：E13-A2/A3 通过 tmux 完成 clip-level teacher signals 与 LOOCV model+rule selector。A3 best **0.6998**，弱超 E12 **0.6831**，但低于 G4/E11 dual fusion **0.7516 ± 0.0946**，不能视为目标完成。
* [x] 2026-07-07：E13-A4/A4b 通过 tmux 完成 MLP lag-corr pair scorer 多 seed。mean train accuracy **0.9888**，eval **0.5989 ± 0.0127** over seeds `0/1/2/3/42/123`，负结果确认。
* [x] 2026-07-07：E13-A5 通过 tmux 完成 window-level temporal selector。移除 invalid GT-derived prototype feature 后，no-leak clip-CV **0.9014**，session-CV **0.8251**，超过 G4/E11 dual fusion mean **0.7516**。下一步若继续，应冻结 A5 feature set 并诊断 `171423_seg1` session-CV failure。
* [x] 2026-07-07：E13-A6 通过 tmux/GPU7 完成 neural MLP window selector 多 seed。clip-CV **0.5908 ± 0.0450**，session-CV **0.5898 ± 0.0068**，负结果；不能用 A6 替代 A5 ridge。
* [x] 2026-07-07：E14-A1/A1b/A2/A3/A4/A5/A6 通过 tmux/GPU 完成 shoulder-local vector 首轮验证。A3 session-heldout **0.5026 ± 0.1250**；A4 augmented original split **0.9005 ± 0.0119** 但已确认泄露；A5 non-leaky augmentation **0.4359 ± 0.1072**，未改善泛化；A6 确认 Protocol A 当前无可用 train clips。
* [x] 2026-07-07：E15-A1/A2 完成 data-integration step。full-session tail 不可用；external supervised custom NPZ 已转换为 E9-style 7D cache，4589 windows。下一步若跑 Protocol A，需要明确接受 `L_LowArm` placement caveat。
* [x] 2026-07-07：E16 MoBind best-config one-session-out 4fold 完成：**0.5998 ± 0.2297**。
* [x] 2026-07-07：E17 topology-aware arm-vector model 完成 A1/A2/A3/A4：preflight 通过，但 full seed0 4fold 只有 **0.3789 ± 0.1760**；targeted ablation 显示 raw IMU tower 最好 **0.5087 ± 0.1215**。下一步若继续，应走 raw 7D IMU + 低容量 skeleton encoder。
* [x] 2026-07-07：E18 source-only realistic EgoHumans pretrain zero-shot session-out 完成：先确认 checkpoint 未见过 custom，再跑 3 source seeds x 4 folds；结果 **0.5009 ± 0.2476**，低于 E16 direct custom **0.5998 ± 0.2297**。
* [x] 2026-07-07：E19 anti-shortcut protocol 源码修正完成：E17 后续训练默认执行 split audit，并支持 label-shuffle / temporal-shift / static controls；E16 四个 one-session-out fold audit 全部通过。
* [x] 2026-07-07：E20 final non-leaky multi-seed comparison 完成：三组方法均为 12/12 结果，A5 summary 已生成；MoBind/E17 realistic transfer 均未超过 direct custom。
* [x] 2026-07-08：E21-A0/A1/A2/A3/A4/A5/A6/A7/A8 已完成。E21 true hybrid source-pretrain model 是当前正结果；6 seeds 为 `0.6550 ± 0.1823`。same-architecture direct custom/no pretrain 为 `0.5120 ± 0.1987`，说明 hybrid alone 不足以解释提升。Full label-shuffle control 为 `0.4628 ± 0.1429`，paired delta CI 为正。E22 暂停 heavy spectral neural branch，除非先设计出能超过 time-domain diagnostic 的 auxiliary/filterbank。
