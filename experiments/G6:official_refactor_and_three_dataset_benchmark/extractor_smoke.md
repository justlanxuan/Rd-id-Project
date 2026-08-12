# G6 三数据集真实 Extractor Smoke

## 目的

证明当前机器上的真实 skeleton backend、依赖、权重和命令仍能工作。该门禁独立于正式训练所复用的历史骨架缓存，不用 cache hit 代替真实推理。

## 固定设置

- backend：`alphapose_full`（YOLO detection + AlphaPose tracking + 2D pose）
- GPU：1
- 每个输入：从对应正式数据源视频隔离生成的 20 帧、10 FPS、640 像素宽短片
- 配置：`configs/g6/extractor_smoke_{totalcapture,egohumans,custom}.yaml`
- 输出根：`/data/fzliang/reid-project/g6/extractor_smoke_outputs`
- 缓存策略：`reuse_existing=false`、`force=true`、`invalid_cache_policy=reextract`
- 成功门槛：非空 AlphaPose JSON、每条有效检测至少 17×3 keypoint、summary 为 `verified_current_run`

## 结果

| 数据集 | 输入 SHA256 | 检测帧 | JSON 条目 | track 数 | cache | provenance |
|---|---|---:|---:|---:|---|---|
| TotalCapture | `aa3d99aba6c063a56c682137823b65c541fe33c05a367361b37cea1396602384` | 20 | 20 | 1 | extracted | verified_current_run |
| EgoHumans | `fa8ea62abebb947bed9e78d9b9183d0bd70a65367c456c4f0862a0bd29edab75` | 20 | 119 | 13 | extracted | verified_current_run |
| Custom | `37f47896e2329ed4048d5357ed89c064bd5561f0e9dd580bcb671d700b1bffda` | 20 | 40 | 2 | extracted | verified_current_run |

对应运行 summary：

- `/data/fzliang/reid-project/g6/extractor_smoke_outputs/totalcapture/totalcapture/pipeline_run_summary.json`
- `/data/fzliang/reid-project/g6/extractor_smoke_outputs/egohumans/egohumans/pipeline_run_summary.json`
- `/data/fzliang/reid-project/g6/extractor_smoke_outputs/custom/custom/pipeline_run_summary.json`

## 失败与修正记录

1. `autism_test` 环境能使用 CUDA，但缺少 `tqdm`，AlphaPose CLI 在 import 阶段失败；未把它记录为后端成功。
2. `/home/fzliang/Autism-project/third-party/AlphaPose` 有检测与姿态权重，但缺 `--pose_track` 所需 OSNet 权重；第一次 TotalCapture 推理显式失败且没有采纳输出。
3. 完整资产位于 `/home/fzliang/repo/AlphaPose`，其专用 venv、YOLO 权重、姿态权重和 OSNet 权重均通过预检；切换后三个数据源均成功。
4. OpenCV 无法初始化 H.264 可视化 writer，但结构化 JSON 正常产生并通过 validator；该问题不影响训练所需 artifact，保留为非阻塞的可视化技术债。
5. 旧 ByteTrack fragment 指向不存在的权重路径；实际权重位于 `/home/fzliang/repo/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar`。本轮没有把 ByteTrack 路径预检当作 ByteTrack 实跑成功。
