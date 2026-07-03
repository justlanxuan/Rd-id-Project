# E6:fair_single_imu_same_window 实时进度日志

## 2026-07-02 17:00
* ✅ 已确认 `MoBind/configs/config.py` 恢复为 5-limb 默认配置。
* ✅ 已清理旧 cache (`cache_action_1.2_0.8_old`) 与被 cache bug 污染的 Stage1/Stage2 checkpoint。
* ✅ 已更新对比图 `results/figures/frameacc_single_imu_same_window.png`。
* ✅ 已更新 G_egohumans 综合对比表 (`results/control_variable_summary.md/json`)。
* ✅ 已连锁修正 E7/E8/E9/plan.md、results.md、progress.md、code_audit.md 与 `formulation.md` 中的旧结论。

## 2026-07-02 16:45
* ✅ E6 cache bug 修复后重训完成（task `bash-s8d221e4`，耗时 7h 1m）。
* ✅ MoBInd single-IMU / 24 帧 **修正后 FrameAcc = 0.9548**（24 sequences）。
* ✅ 结论反转：原始低性能（0.4393 / 0.4556）完全由 cache 污染导致；单 IMU + 24 帧对 MoBInd 完全可行。
* ✅ 已更新 `results/results.md`、`diagnosis.md`、`.log/resume.md`。
* ✅ 已生成修正后的对比表，下一步更新 G_egohumans 汇总表。

## 2026-07-02 09:50
* ✅ 已修复 `MoBind/preprocess/EgoHumans/cache.py`：加入固定肢体顺序映射，按名称取正确通道。
* ✅ 已重新生成 E6 cache，验证前 10 窗口与 raw 数据 **max diff = 0.0**。
* 🔄 Stage1 训练中（task `bash-s8d221e4`），当前约 epoch 26，val R@1 ≈ 2.8%。
* 等待 Stage1 early stop（patience=1000）→ Stage2 → eval。

## 2026-07-02 09:45
* ⚠️ 根因进一步澄清：
  - 原始 `preprocess/EgoHumans/cache.py` 用 `enumerate(limb_list)` 直接索引 `imu_data`。
  - 当临时把 `limb_list` 改为 `["RightWrist"]` 时，实际取到的是索引 0 的 `LeftWrist`，但文件名是 `RightWrist`。
  - 这导致训练/测试数据不一致，是 E6 FrameAcc 异常低的主要嫌疑。
* 已停止之前的错误训练任务，准备启动修正后的 `rerun_correct_cache.sh`。

## 2026-06-25 22:50
* MoBInd Stage2 训练完成（early stopping，约 34 分钟）。
* MoBInd 评估修复：raw IMU 需取 RightWrist 通道后再输入模型。
* MoBInd single-IMU 24-frame mean FrameAcc = **0.4393**。
* 最终对比：
  * MoBInd single IMU (24-frame): **0.4393**
  * Our pipeline single IMU (1 window): **0.7572**
  * Our pipeline single IMU (4-window vote): **0.7973**
* 生成 `results.md` 与对比图。
* 待完成：更新 resume.md、commit/push。

## 2026-06-25 22:10
* Our pipeline 训练完成（50 epochs，best val top1=0.7161）。
* Our pipeline 评估完成：
  * 1-window mean FrameAcc = **0.7572**
  * 4-window vote mean FrameAcc = **0.7973**
* MoBInd Stage1 训练到 epoch 632 被手动截停（val R@1 ≈ 0.219），best checkpoint 已保存。
* 开始 MoBInd Stage2 训练（stage1_exp 指向 Stage1 实际输出目录，epochs=500，patience=100）。

## 2026-06-25 21:55
* Our pipeline 训练已启动（单 IMU 重复 4 次）。

## 2026-06-25 21:43
* Slice 完成（第二次尝试）：123 序列，13435 个 windows。

## 2026-06-25 21:42
* MoBInd 24 帧单 IMU contrastive cache 生成完成：`cache_action_1.2_0.8`，13427 个窗口。

## 2026-06-25 21:30
* Plan 已审批，选择 Option A：统一使用 24 帧窗口（约 1.2 s @ 20 Hz）。
* 初始化 E6 沙盒目录与 plan.md。
* 开始 A1：准备 MoBInd 单 IMU 24 帧配置与代码补丁。

## 2026-07-02 00:40
* ✅ 为与 E8/E9 对齐 Stage2 训练预算（epochs 10000 / patience 500），重新运行 E6 Stage2 + eval。
* 对齐后结果：**MoBInd single-IMU 24-frame mean FrameAcc = 0.4556**
* 与原始 E6（0.4393）几乎一致，说明训练预算不是导致低性能的原因；**单 IMU + 24 帧窗口本身确实很难**。
* 新增文件：
  - `config/MoBind_stage2_w24_aligned.yaml`
  - `scripts/run_aligned_stage2.sh`
  - `results/mobind_single_imu_w24_aligned.json`
* 已更新 `experiments/G_egohumans/results/control_variable_summary.md/json` 与 `.log/resume.md`。

## 2026-07-02 01:25
* ⚠️ 开始按照 HAROS 进行根因排查，撰写 `diagnosis.md`。
* 排查结论：
  - E6 的 contrastive cache (`cache_action_1.2_0.8`) 与当前 `extracted_data` raw `.npy` **存在系统性不一致**（前 10 个窗口最大绝对差 >10）。
  - 作为对照，E8 (`cache_action_5_2`) 和 E9 (`EgoHumans_5imu_w24/cache_action_1.2_0.8`) 的 cache 与 raw **完全一致**（最大绝对差 0.0）。
  - 这意味着 **E6 模型训练用的数据分布与测试时不同**，极可能是导致低性能的真正原因。
* 已启动修复验证：
  - 重新生成 E6 cache（使用当前 `extracted_data`）。
  - 使用对齐的 Stage2 预算重新训练 Stage1 + Stage2。
  - 重新 eval，输出到 `results/mobind_single_imu_w24_correct.json`。
* 运行任务：`bash-1ofnmbwm`，脚本 `scripts/rerun_correct_cache.sh`，日志 `logs/rerun_correct_cache.log`。
