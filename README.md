# Autism Reid Project

IMU-video alignment and re-identification

## Quick Start

### 1. Environment Setup

```bash
# Create base environment
conda create -n autism_test python=3.10 -y
conda activate autism_test

# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install numpy==1.26.0 scipy==1.11.4 pandas matplotlib opencv-python \
    tensorboard timm einops scikit-learn scikit-image pycocotools lap \
    cython-bbox thop easydict loguru natsort tabulate
```


### 2. Prepare External Repositories & Checkpoints

#### 2.1 Clone submodules (AlphaPose + ByteTrack)

This repository uses Git submodules for third-party pose estimation and tracking tools. After cloning, initialize them:

```bash
git submodule update --init --recursive
```

Submodules configured:
- **AlphaPose** → `third-party/AlphaPose` (`https://github.com/L-Ark/AlphaPose.git`)
- **ByteTrack** → `third-party/ByteTrack` (`https://github.com/L-Ark/ByteTrack.git`)

#### 2.2 Clone additional dependencies (MotionBERT + despite)

These repositories are expected to exist as **sibling directories** next to this project (or update the paths in your config accordingly):

```bash
# In the parent directory of this repository
git clone https://github.com/Walter0807/MotionBERT.git
git clone https://github.com/thkreutz/despite.git
```

Expected layout:
```
/workspace/
├── Autism-project/          # this repository
├── MotionBERT/              # pose backbone
└── despite/                 # optional IMU encoder pretraining
```

#### 2.3 Download checkpoints

| Checkpoint | Expected Path | Download Source | Purpose |
|------------|---------------|-----------------|---------|
| **MotionBERT-Lite** | `MotionBERT/checkpoint/pretrain/MB_lite_models.bin` | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS27Ydcbpxlkl0ng?e=rq2Btn) | Pose backbone |
| **AlphaPose** | `Autism-project/third-party/AlphaPose/pretrained_models/fast_res50_256x192.pth` | [Google Drive](https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn) | 2D pose detection |
| **ByteTrack** | `Autism-project/third-party/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar` | [Google Drive](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing) | Person tracking |
| **IMU encoder** | `despite/pretrained_models/v2/SIE_v2.pth` | [TUDataLib](https://tudatalib.ulb.tu-darmstadt.de/items/17c47531-5e6d-4c86-a685-740d8f94f398) (download `OpenAccess_models.zip`) | DeSPITE pre-trained Skeleton+IMU encoder |

> **Note:** If neither IMU checkpoint is available, the matcher will still train from random initialization.


### 3. Run the Full Pipeline

The unified pipeline supports five stages: `preprocess` (raw data alignment), `extract` (video skeleton), `slice` (align + windowing), `train`, `test`.


```bash
# Run everything
./run.sh configs/totalcapture_vicon.yaml all

# Or run individual stages
./run.sh configs/totalcapture_video_test.yaml preprocess   # raw ->  NPZ + video manifest
./run.sh configs/totalcapture_video_test.yaml extract      # video -> skeleton (for video workflows)
./run.sh configs/totalcapture_video.yaml slice        # align extract -> windowed NPZ + csv
./run.sh configs/totalcapture_video.yaml train        # train matcher
./run.sh configs/totalcapture_video.yaml test         # evaluate matcher
```

> **Note:** The default stage order for `all` is `preprocess -> extract -> slice -> train -> test`. `extract` is automatically skipped if no `extract` section is present in the config.

> **IMU filtering:** You can enable an FFT low-pass filter for IMU windows with `imu.lowpass_cutoff_hz` in the config (for example, `20.0`). If your IMU sampling rate is low, the code will clip the cutoff to a safe Nyquist-aware value.

You can also call the Python CLI directly:

```bash
python -m src.pipelines --config configs/totalcapture_video_test.yaml --stages all
python -m src.pipelines --config configs/totalcapture_video_test.yaml --stages extract,slice
```

---

## Available Configs

| Config | Description |
|--------|-------------|
| `configs/totalcapture_vicon_test.yaml` | Quick Vicon test: S1 only, 2 epochs |
| `configs/totalcapture_vicon.yaml` | Full Vicon training: S1-S5, 50 epochs |
| `configs/totalcapture_video_test.yaml` | Video workflow quick test: 1 video, S1, 2 epochs |
| `configs/totalcapture_video.yaml` | Full video workflow training: all videos, S1-S5, 50 epochs |
| `configs/custom.yaml` | Custom 4-fold cross-validation |

---

## Project Structure

```
Autism-project/
├── configs/                      # YAML configuration files
├── src/
│   ├── pipelines/                # High-level workflow orchestration + CLI entry
│   │   ├── __main__.py           # Unified pipeline driver (`python -m src.pipelines`)
│   │   ├── base.py               # PipelineStage base class
│   │   ├── stages.py             # Extract / Slice / Train / Test stages
│   │   ├── full_pipeline.py      # Compose and run stages
│   │   └── video_pipeline/       # Skeleton extraction backends (sub-orchestrators)
│   │       ├── dispatcher.py     # Route to extractor backend by tracker/estimator
│   │       └── backends/
│   │           └── alphapose_bytetrack.py
│   │
│   ├── engine/                   # Training & evaluation engines
│   │   ├── common.py             # Shared model-building utilities
│   │   ├── train.py              # Training script
│   │   ├── eval.py               # Standard evaluation
│   │   ├── eval_custom.py        # 2-person matching evaluation
│   │   ├── eval_grouped.py       # Grouped evaluation
│   │   └── eval_synchronous.py   # Multi-person HOTA/AssA evaluation
│   │
│   ├── datasets/                 # Dataset adapters + PyTorch Datasets
│   │   ├── alignment_dataset.py  # WindowAlignmentDataset
│   │   ├── totalcapture.py       # TotalCapture preprocessing adapter
│   │   ├── custom.py             # Custom 4-fold adapter
│   │   ├── mmact.py              # MMAct adapter (future)
│   │   └── base.py               # BaseData / BaseProcess abstractions
│   │
│   ├── data/                     # Data preparation & slicing tools
│   │   ├── slice/                  # IMU-skeleton slicing entrypoints
│   │   │   ├── totalcapture.py
│   │   │   └── (all datasets now share the same slice entrypoint)
│   │   ├── preprocess/             # Preprocessing helpers
│   │   │   └── totalcapture_unified.py  # Generate unified NPZ for TotalCapture
│   │   └── adapters/
│   │       └── alphapose.py
│   │
│   ├── modules/                  # Core algorithm modules
│   │   ├── encoders/             # IMU / Video encoders
│   │   │   ├── imu.py            # IMUEncoder
│   │   │   ├── video.py          # VideoEncoder
│   │   │   └── utils.py          # MotionBERT backbone builders
│   │   │
│   │   ├── matchers/
│   │   │   ├── base.py
│   │   │   ├── hungarian.py      # Pure-algorithm matcher
│   │   │   ├── dl_matchers/      # Deep-learning matchers
│   │   │   │   ├── imu_video_matcher.py
│   │   │   │   └── despite_matcher.py
│   │   │   ├── physics_matchers/ # Future: physics-based matchers
│   │   │   └── losses.py
│   │   │
│   │   ├── trackers/             # Tracking adapters
│   │   │   ├── base.py
│   │   │   ├── bytetrack.py
│   │   │   └── alphapose.py
│   │   │
│   │   └── pose_estimators/      # Pose-estimation adapters
│   │       ├── base.py
│   │       ├── alphapose.py
│   │       └── wham_3d.py
│   │
│   └── utils/                    # Utilities
│       ├── config.py
│       ├── factory.py            # Lightweight registry
│       ├── chunk_matcher.py
│       └── merge_tracklets.py
│
├── data/
│   ├── processed/                # Preprocessed NPZ + CSV outputs
│   └── interim/                  # Video skeleton extraction outputs (reusable)
├── artifacts/                    # Training checkpoints & logs
├── third-party/                  # AlphaPose, ByteTrack
└── run.sh                        # Bash wrapper around the pipeline
```

---

## Data Preparation & Slicing

### Step 1: Preprocess (unified NPZ generation)

The `preprocess` stage scans raw dataset directories and produces **standardized per-sequence NPZ files**. These NPZs are the canonical data format for the rest of the pipeline.

For **TotalCapture** this includes:
- IMU sensor data
- GT skeleton (from Vicon)
- GT annotations (`person_id`, `bbox`, `visibility`)
- Video path reference
- `video_manifest.csv` (for extract stage)

```bash
python -m src.pipelines --config configs/totalcapture_video_test.yaml --stages preprocess
```

> **Note:** Vicon-based configs also use the same preprocess stage now. It is part of the default `all` stage list.

---

### Step 2: Slice (align + windowing)

The `slice` stage reads the unified NPZs from `preprocess`, optionally aligns extracted video skeletons into them, and slices into windowed training/testing examples.

It produces:
1. **Enriched per-sequence `.npz` files** — original unified NPZ plus `extract_*` arrays and `gt_to_extract_map` (when `skeleton_source=alphapose`).
2. **CSV metadata tables** — sliding-window indices with `skeleton_source`, `person_idx`, `imu_idx` for independent-person training.

#### Running Slice

```bash
# As part of the full pipeline
./run.sh configs/totalcapture_video_test.yaml all

# Or run only the slice stage
./run.sh configs/totalcapture_video_test.yaml slice
```

Under the hood this executes `src.datasets.totalcapture.TotalCaptureAdapter` for all supported datasets.

#### Output Layout

```
data/interim/<project>/slice/
├── sequences/
│   ├── totalcapture_S1_acting1_cam1.npz
│   ├── totalcapture_S1_acting2_cam1.npz
│   └── ...
├── sequences.csv
├── windows_all.csv
├── windows_train.csv
├── windows_val.csv
└── windows_test.csv
```

### Unified NPZ Schema

Each `.npz` under `sequences/` contains the full time-axis data for one sequence:

| Key | Shape | Description |
|-----|-------|-------------|
| `video_path` | scalar (str) | Original video path |
| `dataset` | scalar (str) | Dataset name |
| `sequence_id` | scalar (str) | Unique sequence ID |
| `frame_ids` | `(T,)` | Frame indices aligned to video |
| `imu` | `(T, N_imu, 48)` | IMU features per frame |
| `imu_ids` | `(N_imu,)` | Global IMU / person IDs |
| `gt_person_ids` | `(N_gt,)` | GT person IDs |
| `gt_bboxes` | `(T, N_gt, 4)` | GT bounding boxes `[x1, y1, x2, y2]` |
| `gt_visibility` | `(T, N_gt)` | Bool mask for GT presence |
| `gt_skeleton` | `(T, N_gt, 17, 3)` | GT 3D skeleton (H36M format) |
| `extract_person_ids` | `(N_pred,)` | Extracted track IDs |
| `extract_bboxes` | `(T, N_pred, 4)` | Extracted bboxes |
| `extract_visibility` | `(T, N_pred)` | Extracted presence mask |
| `extract_skeleton` | `(T, N_pred, 17, 3)` | Extracted skeleton |
| `gt_to_extract_map` | `(T, N_gt)` | IoU-based mapping: GT → extract track index (`-1` = unmatched) |

For **single-person** datasets, `N_imu = 1` and `N_gt = 1`. For **multi-person**, all arrays expand naturally along the person dimension.

### CSV Format (`windows_{train,val,test}.csv`)

Each row is one training window:

| Column | Meaning |
|--------|---------|
| `subject` | Subject identifier (e.g. `S1`) |
| `session` | Session / action name (e.g. `acting1`) |
| `split` | `train`, `val`, or `test` |
| `npz_path` | Relative path to the per-sequence NPZ |
| `window_start` | Starting frame index |
| `window_end` | Ending frame index |
| `window_len` | Window length |
| `skeleton_source` | `gt` or `extract` — which skeleton to load for training |
| `person_idx` | Index of the person inside the NPZ |
| `imu_idx` | Index of the IMU inside the NPZ |

`WindowAlignmentDataset` reads the CSV, loads the NPZ on demand, and uses `skeleton_source` + `person_idx` + `imu_idx` to extract the correct `(imu, skeleton)` pair. For `extract` source, missing frames (where `gt_to_extract_map == -1`) are filled with zeros.

### Key Config Options

```yaml
slice:
  window_len: 24
  stride: 16
  train_subjects: S1,S2,S3
  val_subjects: S4
  test_subjects: S5
  max_sequences: 1          # 0 = all sequences
  skeleton_source: alphapose   # auto-derived from extract.pose_estimator
  skeleton_root: ...        # auto-derived from extract.results_root
```

> **Tip:** All paths under `slice` are auto-derived by `resolve_config`. You only need to override them for non-standard layouts.

---

## Workflows

### Vicon Skeleton (Ground Truth)

Uses high-precision motion capture data. **Faster and more accurate.**

Example: `configs/totalcapture_vicon_test.yaml`

```bash
./run.sh configs/totalcapture_vicon_test.yaml all
```

### Video Skeleton Extraction

Extracts skeleton from videos using the detector / tracker / pose-estimator combination specified in your config.

Example: `configs/totalcapture_video_test.yaml`

```yaml
project: totalcapture_video_test

preprocess:
  dataset: totalcapture
  raw_root: /data/fzliang/totalcapture
  camera: cam1
  # output path is auto-derived: data/interim/{project}/preprocess/video_manifest.csv

extract:
  detector: bytetrack
  tracker: bytetrack
  pose_estimator: alphapose
  limit: 1
  skip_existing: true
  gpu: 0
  merge_tracklets:
    enabled: false
    max_gap: 10000000
    score_thresh: 2.2
    max_norm_dist: 2.8
    max_size_diff: 1.8
    fill_gaps: false
    known_num_people: 1

slice:
  window_len: 24
  stride: 16
  sensor_order: [L_LowLeg, R_LowLeg, L_LowArm, R_LowArm]
  train_subjects: S1
  val_subjects: S1
  test_subjects: S1
  max_sequences: 1
  # root, skeleton_source, skeleton_root are auto-derived

train:
  model:
    motionbert_root: /home/fzliang/origin/MotionBERT
    motionbert_config: configs/pose3d/MB_ft_h36m_global_lite.yaml
    motionbert_ckpt: checkpoint/pretrain/MB_lite_models.bin
    imu_ckpt: /home/fzliang/despite/pretrained_models/v2/SIE_v2.pth
  epochs: 2
  batch_size: 32
  num_workers: 4
  compute_imu_stats: true
  imu_sensor: R_LowArm
  repeat_single_sensor: 4

test:
  batch_size: 32
  grouped_test:
    enabled: true
    group_sizes: "2,4,6,8,16"
    num_trials: 50
    chunk_windows: 30
    min_chunk_windows: 15
    seed: 42
  synchronous_test:
    enabled: true
    window_size: 24      # Hungarian matching window size
    stride: 8            # Window step for synchronous eval
```

> **Auto-derived paths:** `resolve_config` creates a stable `work_dir` under
> `data/interim/{project}/` and derives all stage I/O folders from it:
> - `preprocess` outputs → `{work_dir}/preprocess/`
> - `extract` outputs → `{work_dir}/extract/`
> - `slice` outputs → `{work_dir}/slice/`
> - `train` outputs → `{work_dir}/train/`
> - `paths.data_root`, `train_csv`, `val_csv`, `test_csv` are inferred automatically.
> You only need to override them for non-standard layouts.

> **Config fragments:** per-component defaults are loaded automatically from
> `configs/trackers/{name}.yaml` and `configs/pose_estimators/{name}.yaml`.
> You only need to list keys in the workflow YAML when you want to override them.

```bash
./run.sh configs/totalcapture_video_test.yaml all
```

The pipeline will:
1. **Preprocess** — scan raw data, generate unified `npz` + `video_manifest.csv`
2. **Extract** — run video skeleton extraction (ByteTrack + AlphaPose, etc.)
3. **Slice** — align extracted skeletons into the unified NPZs via bbox IoU, then slice into windowed CSVs
4. **Train** — train the IMU-Video matcher in independent-person mode
5. **Test** — run standard eval, grouped eval, and **synchronous multi-person eval** (HOTA/AssA)

---

## Synchronous Multi-Person Evaluation (HOTA / AssA)

When `test.synchronous_test.enabled: true`, the `test` stage runs `eval_synchronous.py`, which evaluates identity association accuracy over full sequences:

1. Load each test sequence NPZ (with `extract_skeleton`, `gt_to_extract_map`, etc.).
2. Slide a temporal window over the full sequence.
3. Within each window, compute embeddings for **all visible extracted tracks** and **all IMUs**.
4. Run **Hungarian matching** per window to assign predicted tracks to IMU IDs.
5. Fuse adjacent window assignments into a per-frame track ID sequence.
6. Format the results as MOT challenge data and compute **HOTA** metrics using `trackeval`.

Key metrics reported:
- **HOTA** — overall tracking accuracy
- **AssA** — association accuracy (how consistent the ID assignments are over time)
- **AssRe / AssPr** — association recall / precision
- **DetA / DetRe / DetPr** — detection accuracy, recall, precision
- **LocA** — localization accuracy

Example output:
```json
{
  "num_sequences": 1,
  "sequences": [
    {
      "sequence_id": "totalcapture_S1_acting1_cam1",
      "HOTA": 1.0,
      "AssA": 1.0,
      "DetA": 1.0,
      ...
    }
  ],
  "mean_HOTA": 1.0,
  "mean_AssA": 1.0,
  ...
}
```

> **Note:** `trackeval` is required. If you encounter NumPy 2.x compatibility issues with PyTorch 2.1, use `numpy<2` and `scipy<1.14` (the package still works despite the pip warning).

---

## Visualization

The `src/visual/visualize_bboxes.py` script overlays bounding boxes and person/track IDs on source videos. It supports GT annotations, AlphaPose tracking results, and side-by-side comparison.

### Supported Inputs

| Source | Required Argument | Description |
|--------|-------------------|-------------|
| **Unified NPZ** | `--input_npz` | Reads `gt_bboxes`, `gt_visibility`, `gt_person_ids`. If `imu_person_map` is present, MAC addresses are shown next to IDs. |
| **Annotation CSV** | `--anno_csv` + `--video` | Reads raw `p{N}_bbox_*` columns. |
| **AlphaPose JSON** | `--alphapose_json` + `--video` | Reads tracked detections (`box` / `bbox` + `idx`). |

### Usage Examples

**Visualize GT from preprocessed NPZ** (auto-fetches `video_path` from NPZ):
```bash
python -m src.visual.visualize_bboxes \
  --input_npz data/interim/custom/preprocess/sequences/custom_20260211_171423.npz \
  --output visual/gt.mp4
```

**Visualize GT from annotation CSV:**
```bash
python -m src.visual.visualize_bboxes \
  --video /data/fzliang/custom/2person/20260211_171423/video/20260211_171423.mp4 \
  --anno_csv /data/fzliang/custom/2person/annotations/20260211_171423.anno.csv \
  --output visual/gt.mp4
```

**Visualize AlphaPose tracking results:**
```bash
python -m src.visual.visualize_bboxes \
  --video /data/fzliang/custom/2person/20260211_171423/video/20260211_171423.mp4 \
  --alphapose_json data/interim/custom_complete/extract/20260211_171423/alphapose_raw/alphapose-results.json \
  --output visual/pred.mp4
```

**Side-by-side GT vs Prediction comparison:**
```bash
python -m src.visual.visualize_bboxes \
  --video /data/fzliang/custom/2person/20260211_171423/video/20260211_171423.mp4 \
  --anno_csv /data/fzliang/custom/2person/annotations/20260211_171423.anno.csv \
  --alphapose_json data/interim/custom_complete/extract/20260211_171423/alphapose_raw/alphapose-results.json \
  --output visual/compare.mp4 \
  --mode compare
```

- **Left panel** (green): `GROUND TRUTH` — GT bboxes with person IDs
- **Right panel** (red): `PREDICTION` — AlphaPose/ByteTrack bboxes with track IDs

### Options

| Flag | Description |
|------|-------------|
| `--output` | **Required.** Output MP4 path. |
| `--mode` | `single` (default) or `compare`. |
| `--fps` | Override output FPS (default = source video FPS). |
| `--no_progress` | Disable frame counter print. |

---

## Notes

- First run computes IMU statistics and saves to `imu_stats.json` for reuse.
- Use `configs/totalcapture_vicon_test.yaml` or `configs/totalcapture_video_test.yaml` for quick testing (~5 minutes).
- Use `configs/totalcapture_vicon.yaml` or `configs/totalcapture_video.yaml` for full training (~2 hours).
- Video extraction is slow (~5-10 min per video) but only needed once; use `skip_existing: true` to skip already-extracted videos within the same run.
