# 🗺️ Project Strategic & Execution Guide

> ⚠️ **写给 AI:** 本文件是该科研课题的最高全局指南，融合了项目背景、科学目标与工程实现。请在 setup 时或项目发生重大方向调整时，协同人类严格更新此文件，确保人机对齐研究大局。

---

## 1. 项目背景与科学痛点 (Project Background)
* **核心科学问题:** 如何在自然、无约束的多人场景中，仅通过可穿戴 IMU 信号与单目视频，实现长期、稳定、跨帧一致的人员重识别（Re-ID）与轨迹关联。
* **研究重大意义:** 为自闭症儿童室内外行为监测提供非侵入式、低成本的自动身份关联方案，避免依赖人工标注或专用动捕设备。
* **现有方法局限:**
  * 在 TotalCapture（动捕 GT）上指标接近 0.98，但在自采 Custom 数据集上骤降至 0.50–0.60。
  * 近期 corruption gradient 实验表明：Custom 的主要瓶颈并非算法、归一化或传感器数量，而是**原始数据中缺乏可区分的人物运动信号**。
  * 静态帧占比过高（如 session 171724 达 75%）、人物动作高度相似、IMU 放置/偏置不一致，导致模型无法学到稳定的身份判别特征。

---

## 2. 整体科学目标 (Overall Goals)
> 对应 `experiments/overall_goals.md` 的宏观愿景。

* **终极里程碑 (Milestone):**
  * 在 Custom 真实场景上，将 frame-level 身份关联准确率从当前 ~0.51（Sliding Window Vote 后处理）提升至可实用的 0.80+。
  * 同时保持 TotalCapture 上 0.95+ 的稳健性，避免过拟合实验室数据。

* **核心评价指标 (Core Metrics):**
  1. `Primary Metric`: Frame-level accuracy / Group accuracy（G2/G4/G6/G8/G16）—— 身份关联正确率。
  2. `Secondary Metric`: HOTA / AssA / DetA —— 同步多目标跟踪评估指标。
  3. `Diagnostic Metric`: 静态帧比例、跨 session IMU 偏置、人物动作可分性 —— 用于判断数据质量是否足够。

---

## 3. 当前活跃的目标/问题分支 (Active Goal & Issue Branches)
> ⚠️ **写给 AI:** 本项目支持多任务并行或多阶段迭代。请在此处维护当前正在推进的试验分支。当开辟新方向时，在此处追加。

### 🎯 活跃分支一：G1 — 真实场景性能瓶颈诊断
* **对应目录:** `experiments/scale_test_corrupt/`
* **核心任务:** 通过 controlled corruption / shuffle / scale 实验，确认 Custom 数据集性能瓶颈是数据内容问题还是算法问题。
* **当前状态:** 已归档。结论：瓶颈是**数据缺乏判别性运动信号**，而非模型或预处理。

### 🎯 活跃分支二：G2 — 数据质量改善与采集协议
* **对应目录:** 待创建
* **核心任务:** 基于 G1 结论，制定新的数据采集协议（增加人物动作差异性、减少静态帧、统一 IMU 佩戴），并设计可量化的数据质量验证指标。
* **当前状态:** 待人类决策是否进入该分支。

### 🎯 活跃分支三：G3 — 算法/后处理改进
* **对应目录:** `experiments/imu_guided_custom_4fold/`、`experiments/custom_normalization_4fold/`
* **核心任务:** 在不改变数据的前提下，探索更强的 embedding、后处理平滑、自适应 decay、多窗口投票等算法改进。
* **当前状态:** 已大量尝试，收益边际递减；当前 SOTA 后处理为 Sliding Window Vote（~51.5%）。

### 🎯 活跃分支四：G3:custom_failure_diagnosis — Custom 失败根因诊断
* **对应目录:** `experiments/G3:custom_failure_diagnosis/`
* **核心任务:** 系统定位 Custom 数据集性能差的根因，按 H1（IMU 佩戴方向）→ H3（骨架提取质量）→ H4（Custom 数据难度）顺序推进。
* **当前状态:** 已进入 E1，准备从 EgoHumans 自提取 AlphaPose 骨架开始验证 H3。

---

## 4. 仓库架构与工程资产 (Repository Architecture)

### 4.1 仓库目录树解构
```text
├── src/                        # 核心模型、算法、网络架构代码
│   ├── pipelines/              # 统一流程编排与 CLI 入口
│   ├── engine/                 # 训练与评估引擎
│   ├── datasets/               # 数据集适配器与 PyTorch Dataset
│   ├── data/                   # 数据预处理与切片工具
│   ├── modules/                # 核心算法模块（encoders / matchers / trackers）
│   └── utils/                  # 配置、工厂、chunk matching 等工具
├── configs/                    # 实验超参数、模型配置控制文件（yaml/json）
├── scripts/                    # 数据预处理、评估、绘图等辅助脚本
├── experiments/                # 科研实验区（按 HAROS 规范维护）
├── artifacts/                  # 训练检查点与日志
├── data/                       # 数据集输出（processed / interim）
├── third-party/                # AlphaPose、ByteTrack 子模块
├── md/                         # 历史实验总结与诊断文档
└── run.sh                      # 流程 bash 包装脚本
```

### 4.2 核心修改区域声明 (Editable Zones)

* **完全开放区 (AI 可自由重构):** `experiments/`、`scripts/`、`md/`、`.runbook/`、`.log/`
* **半开放区 (AI 修改后必须提请人类 Review):** `src/modules/`（核心网络层、matcher、损失函数）、`configs/`（训练配置）
* **严禁修改区 (只读资产):** `third-party/`（子模块代码，除非明确需要升级版本）、已归档的 `artifacts/*` 检查点

---

## 5. 全流程基本运行指令 (Pipeline Commands)

> 记录从零开始跑通核心流程的最小必要指令集，AI 编写具体实验 `plans/` 时必须以此为基准。

### 5.1 数据准备与预处理

```bash
# TotalCapture Vicon 全量训练
./run.sh configs/totalcapture_vicon.yaml all

# Custom 4-fold 训练（需确认数据路径存在）
./run.sh configs/custom.yaml all
```

### 5.2 模型训练与微调

```bash
# 直接调用 Python CLI
python -m src.pipelines --config configs/custom.yaml --stages train

# 仅训练（跳过 extract/slice）
python -m src.pipelines --config configs/totalcapture_vicon.yaml --stages train,test
```

### 5.3 指标评估与推理测试

```bash
# 运行标准评估 + grouped + synchronous
python -m src.pipelines --config configs/custom.yaml --stages test

# 仅同步多目标评估（HOTA/AssA）
python -m src.engine.eval_synchronous --config configs/custom.yaml --checkpoint artifacts/<run_name>/best.pt
```

### 5.4 当前推荐复现（Custom Baseline + Sliding Vote）

```bash
# 1. 训练 baseline
python -m src.pipelines --config configs/custom.yaml --stages train

# 2. 测试时使用 greedy + sliding window vote
# 在 config 中设置：
#   test.matcher.dl_matcher.enabled: true
#   后处理参数（需通过 experiments/imu_guided_custom_4fold/ 中的脚本配置）
# 预期 Frame Acc ≈ 51.5%
```
