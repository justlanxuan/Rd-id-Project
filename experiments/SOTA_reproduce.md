# 🏆 Current Best Results & Reproduction Runbook (当前最优结果与一键复现指南)

> ⚠️ **写给 AI 的铁律：**
> 1. 当且仅当新实验的指标在核心 Metric 上超越此前记录，且接收到人类确认后，方可更新此文件。
> 2. 写入本文件时，**当前工作区（Working Directory）必须是干净的（Clean）**，所有影响结果的修改必须全部提交至 Git。严禁带有未 Commit 的本地修改！
> 3. 更新此文件后，将旧的最优记录整体剪切并归档到末尾的 `## 6. 历史最优演进` 章节中。

## 1. 核心指标看板 (Metrics Board)

### TotalCapture Vicon（实验室动捕场景）
| 评估指标 (Metric) | 上次最优 (Previous) | 当前最优 (Current SOTA) | 提升幅度 (Delta) | 验证时间 (Date) |
| :--- | :--- | :--- | :--- | :--- |
| G2 Acc | — | **0.9800** | — | 2026-06-11 |
| G4 Acc | — | **0.9600** | — | 2026-06-11 |
| G8 Acc | — | **0.9200** | — | 2026-06-11 |

### Custom 真实场景（per-video 7:3 split，单 IMU 24 帧）
| 评估指标 (Metric) | 上次最优 (Previous) | 当前最优 (Current SOTA) | 提升幅度 (Delta) | 验证时间 (Date) |
| :--- | :--- | :--- | :--- | :--- |
| custom test clips FrameAcc (mean ± std, 3 seeds) | — | **0.613 ± 0.010** | — | 2026-06-27 |

**数据集说明:**
* **切分方式：** 每个 custom 视频先按 `segment_frames=1800` 切成 ~1800 帧的小段，每段再按时间 **7:3** 分为 train / test；val 取自每段 train 部分的时间前 10%。
* **输入设置：** 单 IMU `R_LowArm`，`--repeat_single_sensor 4`，24 帧窗口，stride 16。
* **实验来源：** `experiments/G_egohumans/E10b:custom_only_same_split/`
* **旧 setting 说明：** 此前旧 SOTA（Greedy 0.428 / Sliding Window Vote 0.515）基于 cross-session 4-fold，与本 setting 不可直接比较，已归档至第 6 节。

## 2. 代码与资产锚点 (Code & Artifacts Anchor)
* **Git 仓库状态 (Strictly Clean):**
  * **Commit ID:** `48c2ae1`
  * **活跃分支 (Branch):** `egohumans`
* **依赖的权重文件 (Model Checkpoint):**
  * **E10b SOTA 检查点：** `experiments/G_egohumans/E10b:custom_only_same_split/artifacts/E10b_custom_only_seed*/best.pt`
    * *注意：* `experiments/` 目录在 `.gitignore` 中，因此该检查点仅保存在本地，不进入 Git；复现请按第 4 节重新训练。
  * **TotalCapture 最优检查点：** `artifacts/scale_scale_170/best.pt`（对应 corruption gradient 实验中的 clean 基线， captured at commit `756e59d`）

## 3. 核心配置文件快照 (Configuration Snapshot)
> 避免因外部配置文件被覆盖而导致无法复现。请在这里显式贴出本次跑出最优结果的核心超参数。

### E10b custom per-video 7:3 split 配置快照
```yaml
# 数据与切分
project: custom_complete
custom_slice_dir: data/interim/custom_complete/slice/segment_frames_1800
segment_frames: 1800
train_ratio: 0.63   # 每段前 70% 为 train，其中前 10% 作 val
val_ratio: 0.07
test_ratio: 0.30

# 模型输入
imu_sensor: R_LowArm
repeat_single_sensor: 4
window_len: 24
stride: 16

# 训练
epochs: 50
batch_size: 64
num_workers: 4
lr_heads: 1.0e-4
lr_backbone: 1.0e-5
freeze_backbone_epochs: 5
init_alignment_ckpt: data/interim/egohumans_mobind_aligned_single_imu_100/train/egohumans_mobind_aligned_single_imu_100/best.pt

# 外部依赖
motionbert_root: /home/fzliang/origin/MotionBERT
motionbert_config: configs/pose3d/MB_ft_h36m_global_lite.yaml
motionbert_ckpt: checkpoint/pretrain/MB_lite_models.bin
imu_ckpt: /home/fzliang/despite/pretrained_models/v2/SIE_v2.pth
```

### 历史 TotalCapture / custom baseline 配置快照
```yaml
project: custom_complete
slice:
  window_len: 24
  stride: 16
  train_sessions: "20260211_171423,20260211_171724,20260211_172522"
  val_sessions: "20260211_172257"
  test_sessions: "20260211_172257"
train:
  epochs: 50
  batch_size: 64
  num_workers: 8
  imu_sensor: R_LowArm
  repeat_single_sensor: 4
  model:
    motionbert_root: /home/fzliang/origin/MotionBERT
    motionbert_config: configs/pose3d/MB_ft_h36m_global_lite.yaml
    motionbert_ckpt: checkpoint/pretrain/MB_lite_models.bin
    imu_ckpt: /home/fzliang/despite/pretrained_models/v2/SIE_v2.pth
test:
  grouped_test:
    enabled: true
    group_sizes: "2,4,6,8,16"
    num_trials: 200
    chunk_windows: 30
    min_chunk_windows: 15
    seed: 42
```

## 4. 一键复现流程 (One-Click Reproduction Guide)

> 任何人（包括人类、其他智能体或自动化评测 Harness）通过在终端中完全复制并运行以下命令，必须能够 100% 重现上述指标。

### E10b custom SOTA（0.613 ± 0.010）复现
```bash
# 1. 切换并对齐绝对干净的代码现场
git checkout egohumans
git reset --hard 48c2ae1

# 2. 激活环境
conda activate mobind_repro

# 3. 确认外部依赖存在
# MotionBERT: /home/fzliang/origin/MotionBERT
# despite:    /home/fzliang/despite
# 数据:       data/interim/custom_complete/slice/
# E8 初始化检查点: data/interim/egohumans_mobind_aligned_single_imu_100/.../best.pt

# 4. 构建 custom-only CSV（per-video 7:3 split，~1800 帧分段）
bash experiments/G_egohumans/E10b:custom_only_same_split/scripts/A1_build.sh

# 5. 并行训练 3 seeds（0/42/123），需要 3 张 GPU
bash experiments/G_egohumans/E10b:custom_only_same_split/scripts/A2_train_all.sh

# 6. 并行评估 3 seeds
bash experiments/G_egohumans/E10b:custom_only_same_split/scripts/A3_eval_all.sh

# 7. 聚合多 seed 结果
python experiments/G_egohumans/E10b:custom_only_same_split/scripts/A4_aggregate.py

# 期望输出：custom test clips FrameAcc = 0.613 ± 0.010（7 clips）
```

### 历史 TotalCapture / custom baseline 复现（commit 756e59d）
```bash
# 1. 切换到历史 clean 现场
git checkout main
git reset --hard 756e59d

# 2. 激活环境
conda activate test_reid

# 3. 运行完整 pipeline
./run.sh configs/custom.yaml all

# 4. 运行 Greedy + Sliding Window Vote 后处理
python experiments/imu_guided_custom_4fold/eval_greedy_trained_model.py \
  --checkpoint artifacts/custom_batch_20260505_baseline/best.pt \
  --config configs/custom.yaml \
  --decay 0.0 \
  --vote_window 200 \
  --vote_min_freq 0.3
```

## 5. 关键实验环境与约束备注 (Constraints & Notice)

* **算力硬件快照:** 8 × NVIDIA GeForce RTX 4090 D (24GB)；单卡训练 E10b 一个 seed 约 10–30 分钟。
* **复现注意事项:**
  * 外部仓库 `MotionBERT/`、`despite/` 及数据 `data/interim/custom_complete/` 必须可访问。
  * `experiments/` 目录在 `.gitignore` 中，E10/E10b 的实验文档与检查点均不进入 Git，复现必须本地重新训练/评估。
  * 若使用 `skip_existing: true`，请确保 `data/interim/` 中没有残留旧版本的切片结果。
  * 旧 Greedy + Sliding Window Vote 脚本目前位于 `experiments/imu_guided_custom_4fold/`，非项目默认入口；未来应迁移至 `src/` 或 `scripts/` 并提供统一 CLI。

---

## 6. 历史最优演进归档 (Archived SOTA History)

> 格式：[日期] 指标详情, Commit ID, 贡献者。

* *[2026-06-27]* Custom per-video 7:3 split FrameAcc **0.613 ± 0.010** (3 seeds, 7 clips), Commit: `48c2ae1`, 贡献者: AI (Kimi Code CLI)
* *[2026-06-11]* TotalCapture scale_170 clean G2=0.9800, Commit: `756e59d`, 贡献者: AI (Kimi Code CLI)
* *[2026-05-07]* Custom Greedy 0.428 → Sliding Vote 0.515 (cross-session 4-fold), Commit: `1865750`, 贡献者: AI (Kimi Code CLI)
