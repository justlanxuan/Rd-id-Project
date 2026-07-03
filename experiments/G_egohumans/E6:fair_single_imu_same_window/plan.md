# E6: Fair Single-IMU Same-Window Comparison

## 1. 背景与动机
E5 在 MoBInd 官方 action split 上做了公平对比，但仍有两大混淆变量：
- **IMU 数量**：MoBInd 使用 5 个肢体 IMU，我们的 pipeline 使用 4 个肢体 IMU。
- **窗口长度**：MoBInd 训练和同步评估使用 5 秒 / 100 帧窗口；我们的训练与同步评估使用 24 帧窗口。

用户建议控制这两个变量：**双方都只使用右手腕 IMU，并采用相同大小的窗口**，重新训练后再对比 FrameAcc。

## 2. 目标
在 24 个 MoBInd official test 序列（双方都 unseen）上，重新训练并对比：
- **MoBInd**：单 IMU（RightWrist）+ 统一窗口长度。
- **Our pipeline**：单 IMU（`R_LowArm`，即右手腕）重复 4 次以维持 48-D 输入 + 相同统一窗口长度。

监测指标仍为 **synchronous FrameAcc（mean）**，并保留 our pipeline 的 1-window 与 4-window vote 两种推断模式。

## 3. 关键发现（Plan 阶段只读调研）
- **MoBInd**
  - 默认 `window_sec=5`、`stride_sec=2`、`imu_srate=20`，即 100 帧窗口、40 帧步长。
  - `configs/config.py` 的 `limb_list` 控制使用哪些肢体；`data.num_limbs` 需与 `limb_list` 长度一致。
  - `builder/build_model.py` 未把 `window_sec`/`patch_sec` 从 config 传给 `ConvFormer`，若要改窗口长度需补传。
  - Stage1 / Stage2 都使用 `python train_contrastive.py --config ...`。
- **Our pipeline**
  - 单 IMU 通过 `train.imu_sensor: R_LowArm`、`train.repeat_single_sensor: 4` 实现，会自动把单 sensor tile 成 48-D。
  - 训练窗口由 `slice.window_len` 控制；同步评估窗口由 `test.synchronous_test.window_size` 控制。
  - MotionBERT 支持 `T≤243`，因此 100 帧窗口安全；24 帧窗口也安全。
  - E5 的 aligned split（train 98 / val 6 / test 24）可直接复用。

## 4. 可选实验路线

### Option A：使用更小的 24 帧窗口（约 1.2 s @ 20 Hz）
**思路**：把我们现有的短窗口作为统一标准，改造 MoBInd 适配 24 帧。

**MoBInd 改动**：
1. 复制 `configs/EgoHumans/MoBind_stage1.yaml` / `MoBind_stage2.yaml` 为 `MoBind_stage1_w24.yaml` / `MoBind_stage2_w24.yaml`。
2. 设置 `window_sec: 1.2`（24/20），`stride_sec: 0.8`（16/20），保持 `patch_sec: 0.2`（4 帧/patch，6 个 patch）。
3. 设置 `limb_list: ["RightWrist"]`、`num_limbs: 1`。
4. 修改 `builder/build_model.py`，把 config 中的 `window_sec` 与 `patch_sec` 传入 `ConvFormer`，避免模型仍按 5 s 构建 positional embedding。
5. 重新生成 contrastive cache 与 sync cache（若 sync cache 依赖 limb_list）。
6. 重新训练 Stage1 → Stage2。

**Our pipeline 改动**：
1. 复用 E5 已切好的 `window_len=24` CSV（`data/interim/egohumans_mobind_aligned/slice/`）。
2. 新建 config `egohumans_mobind_aligned_single_imu_24.yaml`：
   - `train.imu_sensor: R_LowArm`
   - `train.repeat_single_sensor: 4`
   - 其余与 E5 相同。
3. 重新训练、评估（`window_size=24`、`stride=16`），输出 1-window 与 4-window vote。

**优点**：窗口小，训练/推断更快，符合“优先使用更小窗口”。
**缺点**：MoBInd 需要改模型构建代码并重新训练 Stage1+Stage2，改动较多、调试风险较高。

---

### Option B：使用 MoBInd 默认的 100 帧窗口（5 s @ 20 Hz）
**思路**：保持 MoBInd 原生窗口，把我们的 pipeline 窗口改大到 100 帧。

**MoBInd 改动**：
1. 复制 `configs/EgoHumans/MoBind_stage1.yaml` / `MoBind_stage2.yaml` 为 `MoBind_stage1_single.yaml` / `MoBind_stage2_single.yaml`。
2. 保持 `window_sec: 5`、`stride_sec: 2`、`patch_sec: 0.2`。
3. 设置 `limb_list: ["RightWrist"]`、`num_limbs: 1`。
4. 重新生成 contrastive cache 与 sync cache（因为 motion feature 只取 RightWrist）。
5. 重新训练 Stage1 → Stage2。

**Our pipeline 改动**：
1. 新建 config `egohumans_mobind_aligned_single_imu_100.yaml`：
   - `slice.window_len: 100`
   - `slice.stride: 40`（与 MoBInd stride_sec=2 对应）
   - `train.imu_sensor: R_LowArm`
   - `train.repeat_single_sensor: 4`
   - `test.synchronous_test.window_size: 100`
   - `test.synchronous_test.stride: 40`
2. 重新 slice（生成新的 `windows_*.csv`）、重新计算 IMU stats、重新训练。
3. 评估 our model：1-window 与 4-window vote（若做 vote，可继续 `group_windows=4`，但窗口总时长变长，可视显存调整）。

**优点**：MoBInd 侧无需改模型构建代码，只需改 config 与 cache；实现更简单、可控。
**缺点**：窗口变大，训练/推断变慢；与“优先更小窗口”相反。

## 5. 推荐方案
**推荐 Option B（100 帧窗口）**。原因：
- MoBInd 官方 checkpoint 就是 100 帧训练产物，保持其原生窗口可避免修改 `builder/build_model.py` 等核心代码，减少出错概率。
- 我们的 pipeline 对窗口长度完全兼容（LSTM + MotionBERT 支持变长），只需改 config 重跑 slice/train。
- 虽然窗口更大，但本实验的科学目标是“消除混淆变量”，而不是追求最快训练。Option B 的成功率更高。

如果用户坚持优先更小窗口，可切换至 Option A。

## 6. 实验步骤（以 Option B 为例）

### A1. 准备 MoBInd 单 IMU 配置与数据
- 在 `experiments/G_egohumans/E6:fair_single_imu_same_window/config/mobind/` 下创建 Stage1/Stage2 YAML。
- 修改 `configs/config.py` 的 `limb_list`（或用 patch 脚本在训练前临时替换）。
- 运行 `preprocess/EgoHumans/cache.py` 与 `cache_sync.py` 生成单 IMU cache。
  - ⚠️ 必须确保 `cache.py` 已修复肢体索引映射 bug（使用固定肢体顺序 `['LeftWrist','RightWrist','LeftKnee','RightKnee','Head']` 按名称取索引），否则单 IMU cache 会保存错误肢体。详见 `diagnosis.md`。

### A2. 重新训练 MoBInd
```bash
cd /home/fzliang/MoBind
python train_contrastive.py --config configs/EgoHumans/MoBind_stage1_single.yaml
# 将 Stage2 config 的 model.stage1_exp 指向 Stage1 输出目录
python train_contrastive.py --config configs/EgoHumans/MoBind_stage2_single.yaml
```

### A3. 准备我们的单 IMU 100 帧配置与数据
- 在 `experiments/G_egohumans/E6:fair_single_imu_same_window/config/` 下创建 `egohumans_mobind_aligned_single_imu_100.yaml`。
- 运行 pipeline 的 slice 阶段（可复用 E5 的 `slice.root`）。
- 训练 our model。

### A4. 同步评估
- 用 MoBInd 训练好的 Stage2 exp_dir 运行 `eval_sync_egoh.py --task person`（或 video）。
- 用 our checkpoint 运行 `src.engine.eval_synchronous`，输出 1-window 与 4-window vote。

### A5. 结果汇总
- 运行对比脚本生成表格、图表与 `results.md`。
- 更新 `resume.md`。

## 7. 资源与时耗预估
| 步骤 | Option A（24 帧） | Option B（100 帧） |
|---|---|---|
| MoBInd cache 生成 | ~30 min | ~30 min |
| MoBInd Stage1 训练 | ~2–4 h（窗口小，cache 窗口多，训练量略大） | ~2–4 h |
| MoBInd Stage2 训练 | ~3–6 h | ~3–6 h |
| Our pipeline slice+stats | 复用 E5，<5 min | ~10–20 min |
| Our pipeline 训练 | ~30–60 min | ~1–2 h（窗口大，batch 内序列变长） |
| 评估与汇总 | ~20 min | ~20 min |

## 8. 测试计划（`test/test.md`）
- **单元验证**：
  - MoBInd Stage2 训练日志中 `num_limbs=1` 无 assert 错误。
  - Our pipeline 训练日志中 `imu_sensor=R_LowArm`、`repeat_single_sensor=4` 被正确解析，输入维度保持 48。
- **端到端验证**：
  - 在 1 个 test 序列上先跑通同步 eval，确认 FrameAcc 在合理范围（单 IMU 应低于 4-IMU，但不应崩溃）。
- **通过标准**：
  - 双方均在 24 test 序列上得到非 NaN 的 mean FrameAcc。
  - 定量对比表成功写入 `results.md`。

## 9. 风险与限制
- **单 IMU 重复 vs. 真单 IMU**：Our pipeline 把同一个 sensor 复制 4 次，而 MoBInd 只使用 1 个 limb。虽然输入语义相同，但模型结构不同（our IMU encoder 仍看到 4 个相同 sensor slot），可能引入微弱的正则化/聚合差异。
- **视频编码器差异**：Our video encoder 使用完整骨架，MoBInd motion encoder 只使用对应肢体 motion。这是两条 pipeline 的架构差异，无法通过本实验完全消除。
- **训练随机性**：重新训练双方模型会带来随机波动，建议固定 seed 并至少训练到 val metric 收敛。

## 10. 交付物
- `experiments/G_egohumans/E6:fair_single_imu_same_window/plan.md`
- `experiments/G_egohumans/E6:fair_single_imu_same_window/progress.md`
- `experiments/G_egohumans/E6:fair_single_imu_same_window/config/`
- `experiments/G_egohumans/E6:fair_single_imu_same_window/scripts/`
- `experiments/G_egohumans/E6:fair_single_imu_same_window/results/results.md`
- `experiments/G_egohumans/E6:fair_single_imu_same_window/test/test.md`
- 更新后的 `.log/resume.md`
