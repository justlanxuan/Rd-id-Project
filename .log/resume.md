# 🔄 HAROS Task Resume Node

> ⚠️ **写给 AI:** 本文件是人机协作的唯一状态机断点。在每次迭代结束或实验中断前，必须精确更新以下字段，严禁留空。

## 1. 当前所处阶段 (Current Stage)
* **实验期:** G_egohumans — **E7:mobind_full_setting_reproduce 已完成**
* **当前具体执行任务:** 用 MoBInd 官方原生配置（5 IMU、5 s / 100 帧窗口）重新训练并评估，验证复现流程正确性。
* **当前子任务:** E7 训练、评估与结果汇总均已完成，等待 commit/push。

## 2. 最新成果
### E7: MoBInd 官方全配置复现
* 恢复 MoBInd `configs/config.py` 的 EgoHumans `limb_list` 为官方默认 5 肢体：`["LeftWrist", "RightWrist", "LeftKnee", "RightKnee", "Head"]`。
* 使用 E7 专用 YAML 配置，数据根目录指向 `/data/lyxie/ReID/Data/egohumans`。
* Stage1 训练（early stopping，epoch 5350，耗时 4.37 h），最终 val R@1 ≈ 0.84。
* Stage2 训练（early stopping，epoch 600，耗时 0.53 h），最终 val R@1 ≈ 0.90。
* 在 24 个 MoBInd official test 序列上评估：
  * **E7 重新训练 MoBInd (5 IMU, 100-frame): 0.9675**
  * E5 官方 checkpoint 对比: **0.9666**
  * 差距仅 +0.0009，说明复现流程正确。
* E7 实验沙盒已生成：
  * `experiments/G_egohumans/E7:mobind_full_setting_reproduce/results/results.md`
  * `experiments/G_egohumans/E7:mobind_full_setting_reproduce/results/mobind_retrained_frameacc_aligned_test.json`

### E6 回顾（已结束）
* 在统一单 IMU（右手腕）与统一 24 帧窗口条件下：
  * **MoBInd single IMU (24-frame): 0.4393**
  * **Our pipeline single IMU (1 window): 0.7572**
  * **Our pipeline single IMU (4-window vote): 0.7973**
* E7 的结果证明该差距来自输入设置差异，而非代码 bug。

## 3. 当前阻塞痛点 (Blockers & Issues)
* 无阻塞。
* **已知局限性**（同 E6）：
  * MoBInd 的 IMU encoder 从头训练，而我们的 pipeline 使用预训练 SIE_v2 IMU encoder，数据效率更高。
  * 在单 IMU 设置下，MoBInd 的 `num_limbs=1` 与我们的 `repeat_single_sensor=4` 在模型结构上仍有差异。

## 4. 下一步行动 (Next Actions)
* [ ] 将 E6、E7 实验区文件强制 add 并 push 到 `egohumans` 分支。
* [ ] 在论文/报告中把 E7 作为“MoBInd 复现正确性”证据，E6 作为“控制输入变量后的公平对比”。
