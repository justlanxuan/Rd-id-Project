# 🔄 HAROS Task Resume Node

> ⚠️ **写给 AI:** 本文件是人机协作的唯一状态机断点。在每次迭代结束或实验中断前，必须精确更新以下字段，严禁留空。

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E6:fair_single_imu_same_window 已完成**
* **当前具体执行任务:** 在统一单 IMU（右手腕）与统一 24 帧窗口条件下，重新训练 MoBInd 与我们的 pipeline，并在 24 个 MoBInd official test 序列上对比 FrameAcc。
* **当前子任务:** E6 训练、评估与结果汇总均已完成，等待 commit/push。

## 2. 最新成果
* 已控制两大混淆变量：IMU 数量（单右手腕）与窗口长度（24 帧，约 1.2 s @ 20 Hz）。
* MoBInd 侧修改：
  * `configs/config.py` 的 EgoHumans `limb_list` 改为 `["RightWrist"]`。
  * `builder/build_model.py` 将 config 的 `window_sec` / `patch_sec` 传入 `ConvFormer`。
  * `models/conv_former.py` 用 `round(...)` 避免 1.2/0.2 浮点截断导致 patch 数错误。
  * `preprocess/EgoHumans/cache.py` 与 `cache_multi_person.py` 支持 float 窗口/步长。
  * 重新训练 Stage1（epoch 632 截停）+ Stage2（early stopping，约 34 分钟）。
* Our pipeline 侧：
  * 复用 E5 对齐 split，仅改 `imu_sensor=R_LowArm`、`repeat_single_sensor=4`。
  * 训练 50 epochs，best val top1=0.7161。
* E6 核心结果（24 个 MoBInd official test 序列）：
  * **MoBInd single IMU (24-frame)**: **0.4393**
  * **Our pipeline single IMU (1 window)**: **0.7572**（领先 31.79 pp）
  * **Our pipeline single IMU (4-window vote)**: **0.7973**（领先 35.80 pp）
* E6 实验沙盒已生成：
  * `experiments/G_egohumans/E6:fair_single_imu_same_window/results/results.md`
  * `experiments/G_egohumans/E6:fair_single_imu_same_window/results/figures/frameacc_single_imu_same_window.png`
  * 评估脚本 `A3_eval_mobind_single_imu.py`、`A4_eval_ours_single_imu.py`、`A5_compare_single_imu.py`

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* **值得注意的局限性**：
  * MoBInd 的 IMU encoder 是从头训练，而我们的 pipeline 使用预训练 SIE_v2 IMU encoder，数据效率更高。
  * MoBInd Stage1 在 epoch 632 被手动截停，可能未完全收敛；继续训练或将 Stage1 训练更久，MoBInd 的单 IMU 性能可能提升。
  * Our pipeline 的 `repeat_single_sensor=4` 是把同一 sensor 复制 4 次以维持 48-D 输入，与 MoBInd 的“真单 limb”在模型结构上仍有差异。

## 4. 下一步行动 (Next Actions)
* [ ] 将 E6 实验区文件强制 add 并 push 到 `egohumans` 分支。
* [ ] 在论文/报告中将 E6 作为“控制输入变量后的公平对比”，并明确标注上述局限性。
* [ ] 如需进一步验证，可延长 MoBInd Stage1 训练时间或尝试 100 帧窗口（Option B）作为补充。
