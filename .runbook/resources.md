# 🛠️ Project Hardware & Data Resources

## 1. 硬件算力资源 (Hardware Config)
* **可用 GPU 资源:** 8 × NVIDIA GeForce RTX 4090 D
* **显存上限约束:** 24 GB / 卡
* **当前空闲显存:** 约 3.5–14 GB/卡（部分卡负载较高，运行前请用 `nvidia-smi` 确认）
* **存储空间限制:** `/home` 挂载剩余约 283 GB（总 3.6 TB，已用 92%）
* **建议最大 Batch Size:** 64（Custom 4-fold 已验证），TotalCapture Vicon 完整训练可用 32–64

## 2. 数据资源列表 (Datasets)

### 📦 数据集 A: TotalCapture
* **基本介绍:** 多人动作捕捉数据集，含高精度 Vicon GT 骨架、IMU 传感器数据及视频，用于训练/验证 IMU-Video matcher。
* **存放地点 (绝对路径):** `/data/fzliang/totalcapture`
* **数据格式/划分:** 按 subject (S1–S5) 与 session 划分；Vicon 配置使用 GT 骨架，Video 配置使用 AlphaPose/ByteTrack 提取骨架。

### 📦 数据集 B: Custom 数据集
* **基本介绍:** 自采多人场景数据（3/4/6 人），用于验证模型在真实场景下的跨域泛化性。
* **存放地点 (绝对路径):** `/data/fzliang/custom`
* **数据格式/划分:** 当前主要 session：`20260211_171423`(3人)、`20260211_171724`(4人)、`20260211_172257`(4人)、`20260211_172522`(6人)；按 session 划分 train/val/test。

## 3. 预训练权重与 Checkpoints (Model Weights)
* **权重存放根目录:** `artifacts/`（项目内）、`MotionBERT/checkpoint/pretrain/`、`despite/pretrained_models/v2/`

### 💾 权重资产清单:
1. **`MotionBERT/checkpoint/pretrain/MB_lite_models.bin`**
   * **一句话介绍:** MotionBERT-Lite 官方预训练权重，作为双分支 Video Encoder 的 pose backbone。
2. **`despite/pretrained_models/v2/SIE_v2.pth`**
   * **一句话介绍:** DeSPITE 预训练 Skeleton+IMU encoder (SIE_v2)，用于 IMU 分支初始化。
3. **`artifacts/custom_batch_20260505_baseline/best.pt`**
   * **一句话介绍:** Custom 数据集上当前推荐的 Baseline 检查点（4-fold 训练，用于后续后处理/推理实验）。
4. **`artifacts/scale_scale_2000_shuffled/...`**
   * **一句话介绍:** TotalCapture scale=2000 corruption/shuffle 实验产出的检查点，用于 scale/ablation 测试。

## 4. 环境与激活方式 (Runtime Environment)
* **环境管理工具:** Conda
* **现有环境:** `test_reid`（按 `environment.yml` 创建）
* **依赖安装指令:**
  ```bash
  conda env create -f environment.yml
  conda activate test_reid
  ```
* **外部依赖仓库:**
  * `MotionBERT/`（与项目平级或按 config 指定 `motionbert_root`）
  * `despite/`（与项目平级或按 config 指定 `imu_ckpt` 路径）
  * `third-party/AlphaPose`（git submodule）
  * `third-party/ByteTrack`（git submodule）
