# E7: MoBInd Full-Setting Reproduction Sanity Check

## 1. 背景与动机
原始 E6 报告 MoBInd 在单 IMU + 24 帧窗口下的 FrameAcc（0.4393）显著低于我们的 pipeline，用户怀疑可能是复现方法存在问题。为了排除训练/评估流程的 bug，先用 **MoBInd 官方原生设置**（5 个 IMU、5 秒/100 帧窗口）重新训练一遍 MoBInd，并评估其 FrameAcc。

后续排查发现 E6 的低分实际由 `cache.py` 肢体索引映射 bug 导致，而非训练流程 bug。E7 的结果（0.9675）与官方 checkpoint（0.9666）一致，仍可作为训练/评估流程正确的证据。

## 2. 目标
在 24 个 MoBInd official test 序列上，使用 MoBInd 官方默认配置（5 IMU、100 帧窗口）重新训练 Stage1 + Stage2，并计算 synchronous FrameAcc，与官方 checkpoint（0.9666）对比。

## 3. 关键前提（Plan 阶段只读确认）
- `/data/lyxie/ReID/Data/egohumans/EgoHumans/cache_action_5_2` 已存在（E2 中生成），包含 5-IMU、window_sec=5、stride=2 的 contrastive cache。
- `/data/lyxie/ReID/Data/egohumans/EgoHumans/cache_action_multi_5_2` 与 `cache_sync_action_20_5` 也已存在，但本实验的 FrameAcc 评估不需要它们。
- `configs/EgoHumans/MoBind_stage1.yaml` 和 `MoBind_stage2.yaml` 仍为官方默认配置：
  - `window_sec: 5`, `stride_sec: 2`
  - `multi_sensor: false`（Stage1）/ `true`（Stage2）
  - `num_limbs: 5`（Stage2）
  - `epochs: 10000`，early stopping patience 1000（Stage1）/ 500（Stage2）
- E5 的 `A5_eval_mobind_aligned_test.py` 已能加载任意 Stage2 exp_dir 并计算 FrameAcc；只需将 `--window_size 100` 改为默认即可。

## 4. 实验路线

### A1. 恢复 MoBInd 默认 limb_list
将 `/home/fzliang/MoBind/configs/config.py` 中 EgoHumans 的 `limb_list` 从 `["RightWrist"]` 改回官方默认：
```python
"limb_list": ["LeftWrist", "RightWrist", "LeftKnee", "RightKnee", "Head"]
```

### A2. 重新训练 MoBInd Stage1
```bash
cd /home/fzliang/MoBind
export WANDB_MODE=disabled
python train_contrastive.py --config configs/EgoHumans/MoBind_stage1.yaml
```
- 输出目录：`./outputs/stage1/EgoHumans/<timestamp>/`
- 使用官方 batch_size 1356、epochs 10000、early stopping patience 1000。

### A3. 重新训练 MoBInd Stage2
将 `configs/EgoHumans/MoBind_stage2.yaml` 的 `model.stage1_exp` 指向 A2 的实际输出目录，然后：
```bash
python train_contrastive.py --config configs/EgoHumans/MoBind_stage2.yaml
```
- 输出目录：`./outputs/mae/EgoHumans/<timestamp>/`
- 使用官方 batch_size 128、epochs 10000、early stopping patience 500。

### A4. 评估 FrameAcc
使用 E5 的同步评估脚本，加载 A3 的 Stage2 exp_dir，在 24 个 test 序列上计算 FrameAcc：
```bash
python experiments/G_egohumans/E5:mobind_aligned_splits/scripts/A5_eval_mobind_aligned_test.py \
  --mobind_exp_dir <A3_exp_dir> \
  --output_json experiments/G_egohumans/E7:mobind_full_setting_reproduce/results/mobind_full_setting_frameacc.json
```

### A5. 结果汇总
- 生成 `experiments/G_egohumans/E7:mobind_full_setting_reproduce/results/results.md`。
- 与 E5 官方 checkpoint 结果（0.9666）对比。
- 更新 `.log/resume.md`。

## 5. 资源与时耗预估
| 步骤 | 预估时间 | 资源 |
|---|---|---|
| 恢复 limb_list | <1 min | CPU |
| Stage1 训练 | 2–4 h（取决于 early stopping） | 1 × RTX 4090 |
| Stage2 训练 | 2–4 h | 1 × RTX 4090 |
| FrameAcc 评估 | ~10 min | 1 × RTX 4090 |
| 结果汇总 | ~10 min | CPU |

## 6. 通过标准
- 训练过程无崩溃、无维度错误。
- 重新训练的 MoBInd Stage2 在 24 test 序列上的 mean FrameAcc ≥ 0.90（接近官方 0.9666 即认为复现成功）。
- 若 FrameAcc < 0.80，则需回退并检查代码改动（尤其是 `build_model.py`、`conv_former.py`、`cache.py`）。

## 7. 风险与应对
- **训练时间过长**：官方配置 patience 较大。若超过 6 h 仍未结束，可考虑在后台任务超时前手动截停，并记录当前最佳 checkpoint；但会标注为“预算受限复现”。
- **代码改动影响默认行为**：我们已经将 `window_sec`/`patch_sec` 从 config 传入 `ConvFormer`，官方 config 中这两项与默认值一致，应无影响。`round()` 修改对 5/0.2=25 也无影响。`cache.py` 的 float 参数支持对整数参数无影响。
- **与官方 checkpoint 差距大**：若出现，优先检查 Stage1 是否加载了正确的预训练权重、Stage2 的 `stage1_exp` 是否指向了 A2 目录、以及评估脚本是否使用了 100 帧窗口。

## 8. 交付物
- `experiments/G_egohumans/E7:mobind_full_setting_reproduce/plan.md`
- `experiments/G_egohumans/E7:mobind_full_setting_reproduce/progress.md`
- `experiments/G_egohumans/E7:mobind_full_setting_reproduce/results/results.md`
- 更新后的 `.log/resume.md`
