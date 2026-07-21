# Re-id Project

IMU-video alignment and re-identification

## Quick Start

### 1. Environment Setup

The code has been validated on Python 3.10, PyTorch 2.1, CUDA 11.8. The
recommended setup is to use the repository environment file:

```bash
conda env create -f environment.yml
conda activate test_reid
export PYTHONPATH="$PWD:$PWD/src"
```

On the current development machine, the SOTA reproduction has been run with:

```text
/home/fzliang/miniconda3/envs/mobind_repro/bin/python
```

If you use that existing environment, replace `python` in commands with the
full path above or set:

```bash
export PYTHON_BIN=/home/fzliang/miniconda3/envs/mobind_repro/bin/python
export PYTHONPATH="$PWD:$PWD/src"
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

#### 2.2 External Code

The production training path uses the in-repository hybrid IMU/skeleton
encoders. MotionBERT and DeSPITE are not required for the official pipeline.
They may still be useful for reproducing archived experiments outside the main
code path.

#### 2.3 Download checkpoints

| Checkpoint | Expected Path | Download Source | Purpose |
|------------|---------------|-----------------|---------|
| **AlphaPose** | `third-party/AlphaPose/pretrained_models/fast_res50_256x192.pth` | [Google Drive](https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn) | 2D pose detection |
| **ByteTrack** | `third-party/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar` | [Google Drive](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing) | Person tracking |

The extractor config fragments live in `configs/detectors/`,
`configs/trackers/`, and `configs/pose_estimators/`. Update the paths there if
your AlphaPose or ByteTrack checkout/checkpoint location differs from the
current machine.


### 3. Run the Full Pipeline

The unified pipeline supports four official stages: `extract` (optional skeleton
extraction), `prepare` (preprocess + slice), `train`, and `evaluate`.


```bash
# Run everything
./run.sh configs/totalcapture_vicon.yaml all

# Or run individual stages
./run.sh configs/totalcapture_video_test.yaml extract     # optional video -> skeleton
./run.sh configs/totalcapture_video_test.yaml prepare     # raw/IMU/skeleton -> NPZ + window CSV
./run.sh configs/totalcapture_video.yaml train            # train matcher
./run.sh configs/totalcapture_video.yaml evaluate         # evaluate matcher
```

> **Note:** The default stage order for `all` is `extract -> prepare -> train -> evaluate`. `extract` is automatically skipped if no `extract` section is present in the config. Legacy stage names `preprocess`, `slice`, and `test` are accepted only as compatibility aliases.

> **IMU filtering:** You can enable an FFT low-pass filter for IMU windows with `imu.lowpass_cutoff_hz` in the config (for example, `20.0`). If your IMU sampling rate is low, the code will clip the cutoff to a safe Nyquist-aware value.

You can also call the Python CLI directly:

```bash
python -m src.pipeline --config configs/totalcapture_video_test.yaml --stages all
python -m src.pipeline --config configs/totalcapture_video_test.yaml --stages extract,prepare
```

---

## Official Pipeline

The codebase is organized around one config-driven workflow. Dataset modules
only describe how to read dataset-specific raw files; shared behavior lives in
`src/preprocess/common`, `src/datasets`, `src/modules`, and `src/engine`.

### Stage Order

```text
extract(optional) -> prepare(preprocess + pack + slice) -> train -> evaluate
```

`extract` is optional. It runs only when the YAML file contains an explicit
`extract:` section. Workflows using ground-truth skeletons, or EgoHumans'
bundled pose files, skip this stage.

### 1. Extract

Purpose: infer skeletons from raw video.

Dataset-specific code is responsible for finding videos and writing a manifest.
The actual extraction implementation is shared:

```text
src.preprocess.common.extract.run_video_skeleton_extraction
```

The CLI wrapper is:

```bash
python src/pipelines/video_pipeline/dispatcher.py --config <config.yaml>
```

The shared extractor reads `extract.video` or `extract.manifest_csv`, assembles
detector/tracker/pose-estimator settings from config fragments, and dispatches
to the configured backend:

- ByteTrack + AlphaPose
- AlphaPose full pipeline
- WHAM

Manifest rows may include `sequence_id` or `result_name`; this controls the
output directory name under `extract.results_root`, so each dataset can keep
stable sequence IDs while reusing the same extraction code.

### 2. Prepare

Purpose: create canonical data for training.

For most users this is one stage:

```bash
python -m src.pipeline --config <config.yaml> --stages prepare
```

Internally, dataset entrypoints split it into:

```text
preprocess -> pack -> slice
```

`preprocess` reads raw dataset files and applies dataset-specific alignment:

- EgoHumans reads extracted `.npy` arrays and converts IMU to 7D `acc3 + quat4`.
- Custom reads raw IMU CSV, converts to 7D, optionally low-pass filters, then
  resamples IMU to video/annotation timestamps.
- TotalCapture reads Xsens/Vicon files and writes unified sequence NPZs.

`pack` groups processed per-person files into unified per-sequence NPZs when a
dataset needs that intermediate step. EgoHumans uses this explicitly.

`slice` writes `windows_train.csv`, `windows_val.csv`, and `windows_test.csv`.
All training and standard evaluation use these CSV files through
`src.datasets.alignment.WindowAlignmentDataset`.

### 3. Train

Training is handled by:

```bash
python -m src.engine.train --config <train_config.yaml>
```

The trainer builds the model via `src.engine.common`, reads windows through
`WindowAlignmentDataset`, fits or loads IMU statistics when requested, and saves
checkpoints under the configured output directory.

The official model path is the hybrid IMU/skeleton matcher:

```text
src/modules/encoders/hybrid.py
src/modules/matchers/dl_matchers/imu_video_matcher.py
```

### 4. Evaluate

Evaluation is handled by:

```bash
python -m src.engine.evaluate --config <eval_config.yaml>
```

Official metrics:

- FrameAcc: window-level or segment-level IMU/skeleton assignment accuracy.
- Group Test: sampled group retrieval/matching accuracy.

The SOTA benchmark uses segment-level FrameAcc on Custom transfer folds.

### IMU Handling

Shared IMU utilities live in:

```text
src/preprocess/common/imu.py
```

Important functions:

- `parse_imu_csv`: read Custom raw IMU CSV.
- `convert_single_imu_to_7d`: build `acc3 + quat4`.
- `lowpass_filter_fft`: FFT low-pass filter.
- `resample_imu_to_target`: align IMU to frame timestamps.
- `quat_to_rotmat` / `rotmat_to_quat_wxyz`: representation conversion.

Training/evaluation-time IMU transforms happen in:

```text
src/datasets/alignment.py
```

That dataset loads `imu` from NPZ/window rows, optionally selects a sensor,
applies low-pass filtering, and applies global or per-session normalization.

### EgoHumans Pipeline

EgoHumans supports three skeleton sources:

1. **Vicon/GT skeleton**: used by the current source pretraining and SOTA
   benchmark. No `extract` stage is needed.
2. **Bundled EgoHumans pose files**: set `preprocess.skeleton_source: pose2d`.
   This conversion belongs to `preprocess`, not `extract`, because no model
   inference is run.
3. **Self-extracted video skeleton**: add an explicit `extract:` section. The
   EgoHumans dataset code writes a video manifest, then calls the shared common
   extraction runner.

EgoHumans stage commands:

```bash
python -m src.preprocess.datasets.egohumans --task preprocess --config <config.yaml>
python -m src.preprocess.datasets.egohumans --task pack --config <config.yaml>
python -m src.preprocess.datasets.egohumans --task slice --config <config.yaml>
```

Use `--task extract` only when the config has an explicit `extract:` section.

### Custom Pipeline

Custom SOTA preparation uses:

```bash
python -m src.preprocess.datasets.custom --config <target_prepare.yaml>
python src/pipelines/video_pipeline/dispatcher.py --config <target_prepare.yaml>
python -m src.preprocess.datasets.custom --task pack_segments --config <target_prepare.yaml>
python -m src.preprocess.datasets.custom --task slice --config <target_prepare.yaml>
```

The first step reads raw video, annotations, and IMU CSV files. The second step
uses the shared skeleton extraction runner. `pack_segments` aligns extracted
skeletons with Custom segments. `slice` creates session-out fold caches.

---

## SOTA Reproduction

The official cross-dataset transfer benchmark is:

```text
configs/benchmarks/cross_dataset_transfer_sota.yaml
```

Run the complete benchmark with one command:

```bash
python tools/run_benchmark.py \
  --config configs/benchmarks/cross_dataset_transfer_sota.yaml
```

The config contains all paths, seeds, fold definitions, device scheduling, and
expected metrics. No files under `experiments/` are needed.

### Environment for the Current SOTA

The current SOTA config assumes the following runtime environment:

```text
OS/GPU: Linux with CUDA GPUs
Python: 3.10
PyTorch: 2.1.x with CUDA 11.8
Validated Python: /home/fzliang/miniconda3/envs/mobind_repro/bin/python
Repository: /home/fzliang/workspace/Re-id-Project
```

Before running, activate a compatible environment and set `PYTHONPATH`:

```bash
cd /home/fzliang/workspace/Re-id-Project
conda activate mobind_repro
export PYTHONPATH="$PWD:$PWD/src"
```

Required raw/intermediate data paths for the checked-in SOTA config:

```text
EgoHumans raw root:
  /data/lyxie/ReID/Data/egohumans

EgoHumans realistic IMU arrays:
  /data/lyxie/ReID_imu_generation/outputs/egohumans_imu_realistic/extracted_data

Custom raw root:
  /data/fzliang/custom

Custom prepared/intermediate outputs:
  /data/fzliang/reid_project/interim/custom_preprocess
  /data/fzliang/reid_project/interim/custom_skeleton
  /data/fzliang/reid_project/interim/custom_segments
  /data/fzliang/reid_project/interim/custom_extracted_data

Benchmark output root:
  /data/fzliang/reid-project/benchmarks/cross_dataset_transfer_sota_e21_compat
```

The benchmark config schedules training on:

```yaml
devices: [cuda:4, cuda:5, cuda:6, cuda:7]
max_parallel: 4
```

Change these fields in `configs/benchmarks/cross_dataset_transfer_sota.yaml`
if your machine uses different visible GPU IDs or fewer GPUs.

Skeleton extraction for Custom requires working AlphaPose and ByteTrack
checkouts/checkpoints. The relevant path fragments are:

```text
configs/trackers/bytetrack.yaml
configs/pose_estimators/alphapose.yaml
configs/detectors/yolox.yaml
```

If the skeleton and segment caches already exist, the benchmark skips expensive
preparation steps according to `skip_existing: true`.

Recommended preflight:

```bash
python tools/run_benchmark.py \
  --config configs/benchmarks/cross_dataset_transfer_sota.yaml \
  --dry-run
```

The dry-run should finish without missing-input errors before launching full
training.

### What It Runs

The default action is `run_all`, equivalent to:

```text
prepare -> check_inputs -> generate -> train_source -> train_target -> evaluate -> summarize
```

Source preparation:

```bash
python -m src.preprocess.datasets.egohumans --task preprocess --config <generated/source_prepare.yaml>
python -m src.preprocess.datasets.egohumans --task pack --config <generated/source_prepare.yaml>
python -m src.preprocess.datasets.egohumans --task slice --config <generated/source_prepare.yaml>
```

Target preparation:

```bash
python -m src.preprocess.datasets.custom --config <generated/target_prepare.yaml>
python src/pipelines/video_pipeline/dispatcher.py --config <generated/target_prepare.yaml>
python -m src.preprocess.datasets.custom --task pack_segments --config <generated/target_prepare.yaml>
python -m src.preprocess.datasets.custom --task slice --config <generated/target_prepare.yaml>
```

Training and evaluation:

```bash
python -m src.engine.train --config <generated/source_train_seed.yaml>
python -m src.engine.train --config <generated/target_fold_seed.yaml>
python -m src.engine.evaluate --config <generated/eval_fold_seed.yaml>
```

### Benchmark Design

- Source dataset: EgoHumans realistic IMU + GT/Vicon skeleton.
- Target dataset: Custom raw video + raw IMU + extracted skeleton.
- Seeds: `0, 1, 2, 3, 42, 123`.
- Target folds: four leave-one-session-out Custom folds.
- Main runs: `4 folds x 6 seeds = 24`.
- Control runs: label-shuffle control, also `24` runs when enabled.
- Multi-GPU scheduling is controlled by:

```yaml
runner:
  devices: [cuda:4, cuda:5, cuda:6, cuda:7]
  max_parallel: 4
```

Expected strict-condition result:

```yaml
expected:
  mean: 0.6550
  std: 0.1823
  runs: 24
  control:
    name: label_shuffle
    mean: 0.4628
    std: 0.1429
```

Useful runner modes:

```bash
# Print/check the workflow without launching training.
python tools/run_benchmark.py --config configs/benchmarks/cross_dataset_transfer_sota.yaml --dry-run

# Only regenerate configs.
python tools/run_benchmark.py --config configs/benchmarks/cross_dataset_transfer_sota.yaml --generate

# Only prepare data.
python tools/run_benchmark.py --config configs/benchmarks/cross_dataset_transfer_sota.yaml --prepare
```

Generated configs, logs, checkpoints, evaluations, and summaries are written
under `runner.output_root` from the benchmark config.

---

## Development Guide

Keep new code aligned with the current official structure:

```text
configs/                  YAML workflow configs and component fragments
src/config/               YACS defaults, compatibility loading, path resolving
src/preprocess/common/    Shared preprocessing, IMU, skeleton, packing, slicing
src/preprocess/datasets/  Dataset-specific raw-data adapters
src/datasets/             PyTorch Dataset and window transforms
src/modules/encoders/     Model encoders
src/modules/matchers/     Matching heads and matching losses
src/modules/trackers/     Tracker adapters
src/modules/pose_estimators/ Pose-estimator adapters
src/engine/               Train/evaluate engines
tools/                    Reproducible benchmark and maintenance CLIs
```

### Adding a New Dataset

Add one file under:

```text
src/preprocess/datasets/<dataset>.py
```

The dataset module should only handle dataset-specific concerns:

- discover raw files;
- parse dataset metadata;
- align raw IMU/video/skeleton timestamps;
- write canonical sequence NPZs;
- write or call the shared slice logic.

Reuse shared code wherever possible:

- IMU parsing/filtering/resampling: `src/preprocess/common/imu.py`
- video manifest writing: `src/preprocess/common/video.py`
- skeleton extraction: `src/preprocess/common/extract.py`
- AlphaPose JSON loading/alignment: `src/preprocess/common/alphapose.py`
- window slicing: `src/preprocess/common/slice.py`

Then add config defaults only if the setting is broadly reusable:

```text
src/config/defaults.py
src/config/config.py
```

### Adding a New Preprocessing Method

Put shared algorithms in `src/preprocess/common/`. Put only dataset-specific
file discovery or format quirks in `src/preprocess/datasets/`.

Examples:

- New IMU filter: add to `src/preprocess/common/imu.py`.
- New skeleton JSON parser: add to `src/preprocess/common/`.
- New split policy: add to `src/preprocess/common/slice.py` if it is general,
  or to the dataset entrypoint if it is dataset-specific.
- New segment packing logic: prefer a shared helper first; keep dataset-specific
  naming and path conventions in the dataset module.

### Adding a New Encoder

Add model code under:

```text
src/modules/encoders/
```

Then register/build it through:

```text
src/engine/common.py
```

If the encoder needs new config fields, add defaults in:

```text
src/config/defaults.py
```

Do not add long CLI argument lists for model options. Training and evaluation
should remain config-driven.

### Adding a New Matcher or Loss

Use:

```text
src/modules/matchers/
src/modules/matchers/losses.py
src/engine/losses.py
```

Keep pure matching algorithms independent of training code. Training-specific
loss composition belongs in `src/engine/losses.py`.

### Adding a New Skeleton Extraction Backend

Add the adapter under:

```text
src/pipelines/video_pipeline/video_extractors/
```

Then update:

```text
src/preprocess/common/extract.py
configs/trackers/
configs/pose_estimators/
configs/detectors/
```

Dataset modules should not instantiate backend-specific classes directly. They
should write manifests and call the shared extraction runner.

### Adding a New Benchmark

Add a config under:

```text
configs/benchmarks/
```

If the benchmark is structurally similar to cross-dataset transfer, extend
`tools/run_benchmark.py`. Keep experiment-only scripts out of the main
reproduction path.

### Code Review Rules

- Keep official code paths under `src/`, `configs/`, `tools/`, and `docs/`.
- Keep archived exploratory work under `experiments/`; do not make official
  reproducibility depend on it.
- Prefer common helpers over copying logic into a dataset module.
- Preserve the canonical NPZ/CSV schema used by `WindowAlignmentDataset`.
- Add focused tests or dry-runs when changing preprocess, slicing, training, or
  evaluation behavior.

---

## Available Configs

| Config | Description |
|--------|-------------|
| `configs/totalcapture_vicon_test.yaml` | Quick Vicon test: S1 only, 2 epochs |
| `configs/totalcapture_vicon.yaml` | Full Vicon training: S1-S5, 50 epochs |
| `configs/totalcapture_video_test.yaml` | Video workflow quick test: 1 video, S1, 2 epochs |
| `configs/totalcapture_video.yaml` | Full video workflow training: all videos, S1-S5, 50 epochs |
| `configs/egohumans_test.yaml` | EgoHumans raw-first quick test |
| `configs/egohumans.yaml` | EgoHumans raw-first training |
| `configs/egohumans_full.yaml` | EgoHumans larger raw-first training |
| `configs/custom.yaml` | Custom 4-fold cross-validation |

---

## Project Structure

```
Re-id-Project/
├── configs/                      # YAML configuration files
│   ├── benchmarks/               # Reproducible benchmark configs
│   ├── detectors/                # Extract-stage detector fragments
│   ├── trackers/                 # Extract-stage tracker fragments
│   └── pose_estimators/          # Extract-stage pose-estimator fragments
├── src/
│   ├── pipeline.py               # Unified workflow driver (`python -m src.pipeline`)
│   ├── pipelines/                # Video extraction internals + legacy wrappers
│   │   ├── __main__.py           # Compatibility wrapper for old commands
│   │   ├── full_pipeline.py      # Compatibility wrapper around src.pipeline
│   │   └── video_pipeline/
│   │       ├── dispatcher.py     # CLI wrapper around common extraction runner
│   │       └── video_extractors/ # ByteTrack/AlphaPose/WHAM backend adapters
│   │
│   ├── engine/                   # Training & evaluation engines
│   │   ├── augmentation.py       # Training-time input augmentation helpers
│   │   ├── batch.py              # Batch/device and metadata label helpers
│   │   ├── common.py             # Shared model-building utilities
│   │   ├── losses.py             # Training-specific loss composition
│   │   ├── stats.py              # IMU/model-stat fitting helpers
│   │   ├── train.py              # Training script
│   │   ├── validation.py         # Validation loop helpers used by training
│   │   └── evaluate.py           # Official FrameAcc + Group Test evaluator
│   │
│   ├── datasets/                 # Dataset adapters + PyTorch Datasets
│   │   ├── alignment.py          # WindowAlignmentDataset over standardized NPZ/CSV
│   │   ├── alignment_dataset.py  # Backward-compatible import shim
│   │   ├── samplers.py           # Dataset-aware batch samplers
│   │   └── transforms.py         # Dataset-side transforms and legacy adapters
│   │
│   ├── config/                   # YACS workflow configuration
│   │   ├── defaults.py           # Central defaults
│   │   └── config.py             # YAML loader + path resolver
│   │
│   ├── preprocess/               # Raw datasets -> standardized NPZ/CSV
│   │   ├── common/               # Shared IMU, extraction, packing, slicing helpers
│   │   │   ├── extract.py        # Shared video skeleton extraction runner
│   │   │   ├── imu.py            # Shared IMU parsing/filtering/resampling
│   │   │   └── slice.py          # Shared window slicing
│   │   └── datasets/             # Dataset-specific preprocess/slice entrypoints
│   │
│   ├── modules/                  # Core algorithm modules
│   │   ├── encoders/             # Official paired IMU / Video encoders
│   │   │   └── hybrid.py         # HybridIMUEncoder + HybridSkeletonEncoder
│   │   │
│   │   ├── matchers/
│   │   │   ├── base.py
│   │   │   ├── hungarian.py      # Pure-algorithm matcher
│   │   │   ├── dl_matchers/      # Deep-learning matchers
│   │   │   │   └── imu_video_matcher.py
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
├── tools/
│   └── run_benchmark.py          # Official SOTA benchmark runner
├── docs/                         # Additional data/benchmark/development docs
├── experiments/                  # Archived exploratory experiments only
├── third-party/                  # AlphaPose, ByteTrack
└── run.sh                        # Bash wrapper around the pipeline
```

---

## Data Preparation & Slicing

### Data Home

All reusable data and intermediate outputs should live under
`/data/fzliang/reid-project`, not under the repository or `/home/fzliang/...`.

Default layout:

```text
/data/fzliang/reid-project/
├── totalcapture/
│   ├── skeleton/
│   │   └── alphapose/
│   ├── imu/
│   │   └── synthetic/
│   ├── preprocessed/
│   │   └── <project>/
│   └── artifacts/
│       └── <project>/
├── egohumans/
│   ├── skeleton/
│   │   └── alphapose/
│   ├── imu/
│   │   ├── realistic/
│   │   └── mobind/
│   ├── preprocessed/
│   │   └── <project>/
│   └── artifacts/
│       └── <project>/
└── custom/
    ├── skeleton/
    ├── imu/
    ├── preprocessed/
    └── artifacts/
```

`src/config/defaults.py` sets `PATHS.DATA_HOME` to
`/data/fzliang/reid-project`. You can override it without editing YAML:

```bash
export REID_DATA_HOME=/data/fzliang/reid-project
```

New configs should use `raw_root` only for read-only source data. Legacy
relative output paths such as `data/interim/...`, `./data/interim/...`, and
`artifacts/...` are still normalized by the config loader for compatibility,
but should not be used in new YAML. Skeleton extraction outputs go under
`<dataset>/skeleton/<extractor>`. Generated or reusable IMU sources go under
`<dataset>/imu/<source>`. Unified sequence NPZs and window CSVs go under
`<dataset>/preprocessed/<project>`. Checkpoints and evaluation outputs go under
`<dataset>/artifacts/<project>`.

The repository no longer keeps one-off migration scripts. New intermediate data
should be written directly under `/data/fzliang/reid-project`.

### Step 1: Extract Skeletons (optional)

The `extract` stage runs or imports skeleton extraction results and stores them
under `<dataset>/skeleton/<extractor>`, for example:

```text
/data/fzliang/reid-project/egohumans/skeleton/alphapose/
```

If a dataset workflow uses ground-truth skeletons, this stage is skipped. If the
config uses `extract.copy_from`, existing skeleton results are copied into the
standard location.

### Step 2: Prepare (unified NPZ + window CSV)

The `prepare` stage combines the old preprocess and slice responsibilities:

1. Convert raw/read-only dataset sources into standardized per-sequence NPZ
   files under `<dataset>/preprocessed/<project>/sequences`.
2. Write `windows_train.csv`, `windows_val.csv`, and `windows_test.csv` under
   the same `<dataset>/preprocessed/<project>` directory.

These NPZs and CSVs are the canonical data format for training and evaluation.

For **TotalCapture** this includes:
- IMU sensor data
- GT skeleton (from Vicon)
- GT annotations (`person_id`, `bbox`, `visibility`)
- Video path reference
- `video_manifest.csv` (for extract stage)

For **EgoHumans**, the default preprocess path is raw-first and uses the
realistic synthetic IMU source by default. Configs should point to the original
EgoHumans root for skeleton/metadata and to the realistic extracted IMU arrays:

```yaml
preprocess:
  dataset: egohumans
  raw_root: /data/lyxie/ReID/Data/egohumans
  imu_source: realistic
  extracted_root: /data/lyxie/ReID_imu_generation/outputs/egohumans_imu_realistic/extracted_data
```

The official EgoHumans data path is split into explicit stages:
`extract`, `preprocess`, `pack`, and `slice`. `extract` is only for true
video skeleton extraction. Converting EgoHumans' bundled `processed_data/poses2d`
files is part of `preprocess`, so that path skips `extract`.
Set `imu_source: mobind` and `extracted_root:
/data/lyxie/ReID/Data/egohumans/extracted_data` only when intentionally using
the older MoBind-style EgoHumans IMU source.

Use the bundled EgoHumans pose files as extracted skeletons:

```yaml
preprocess:
  dataset: egohumans
  skeleton_source: pose2d

slice:
  skeleton_source: alphapose
```

Run the stages directly when debugging data preparation:

```bash
python -m src.preprocess.datasets.egohumans --task preprocess --config configs/egohumans_test.yaml
python -m src.preprocess.datasets.egohumans --task pack --config configs/egohumans_test.yaml
python -m src.preprocess.datasets.egohumans --task slice --config configs/egohumans_test.yaml
```

Run `extract` first only when the config has an explicit `extract:` section and
you want to infer skeletons from video with the shared video pipeline.

For **custom+**, use `PREPROCESS.DATASET: custom_plus` or `custom+`; both route
to `src.preprocess.datasets.custom_plus`.

```bash
python -m src.pipeline --config configs/totalcapture_video_test.yaml --stages prepare
python -m src.pipeline --config configs/egohumans_test.yaml --stages prepare
```

> **Note:** Legacy commands may still use `--stages preprocess` or
> `--stages slice`; official configs should use `prepare`.

The prepare stage does not duplicate or reinterpret preprocessed sequence NPZs
for GT/Vicon workflows.

It produces:
1. **CSV metadata tables** — sliding-window indices with `skeleton_source`,
   `person_idx`, and `imu_idx` for independent-person training/evaluation.
2. **Aligned sequence NPZs only for extracted-skeleton workflows** — when
   `skeleton_source=alphapose`, extracted skeleton arrays and
   `gt_to_extract_map` are materialized under the slice output directory.

#### Running Slice

```bash
# As part of the full pipeline
./run.sh configs/totalcapture_video_test.yaml all

# Or run only data preparation
./run.sh configs/totalcapture_video_test.yaml prepare
```

Under the hood this executes dataset-specific preprocess entrypoints that emit
the same window CSV schema. Dataset modules live in `src/preprocess/datasets/`
and reuse shared slicing logic from `src/preprocess/common/slice.py`.

#### Output Layout

```
/data/fzliang/reid-project/<dataset>/preprocessed/<project>/
├── aligned_sequences/           # only when skeleton_source=alphapose
│   ├── totalcapture_S1_acting1_cam1.npz
│   ├── totalcapture_S1_acting2_cam1.npz
│   └── ...
├── sequences/
│   ├── totalcapture_S1_acting1_cam1.npz
│   └── ...
├── sequences.csv
├── windows_all.csv
├── windows_train.csv
├── windows_val.csv
└── windows_test.csv
```

### Unified NPZ Schema

Each preprocessed `.npz` under the preprocess output contains the full time-axis
data for one sequence:

| Key | Shape | Description |
|-----|-------|-------------|
| `video_path` | scalar (str) | Original video path |
| `dataset` | scalar (str) | Dataset name |
| `sequence_id` | scalar (str) | Unique sequence ID |
| `frame_ids` | `(T,)` | Frame indices aligned to video |
| `imu` | `(T, N_imu, D)` | IMU features per frame. Current hybrid/SOTA path uses 7D `acc3 + quat4`; legacy/expanded paths may use 48D. |
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

`WindowAlignmentDataset` reads the CSV, loads the NPZ on demand, and uses
`skeleton_source` + `person_idx` + `imu_idx` to extract the correct
`(imu, skeleton)` pair. For `extract` source, missing frames (where
`gt_to_extract_map == -1`) are filled with zeros.

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
  # prepare output is auto-derived:
  # /data/fzliang/reid-project/totalcapture/preprocessed/{project}/

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
    type: hybrid
  epochs: 2
  batch_size: 32
  num_workers: 4
  compute_imu_stats: true
  imu_sensor: R_LowArm
  repeat_single_sensor: 4

test:
  batch_size: 32
  metrics:
    frame_acc:
      enabled: true
    group_test:
      enabled: true
      group_sizes: "2,4,6,8,16"
      num_trials: 50
      chunk_windows: 30
      min_chunk_windows: 15
      seed: 42
```

> **YACS config defaults:** Stable workflow defaults live in
> `src/config/defaults.py`. YAML files only need to contain dataset-specific
> paths, split definitions, run names, and intentional overrides.
> `src/config/config.py` loads YAML through YACS, normalizes legacy lowercase
> keys, and derives paths. `src/utils/config.py` remains as a compatibility
> wrapper for older imports.
> Training and evaluation engines are config-driven: use
> `python -m src.engine.train --config <workflow.yaml>` and
> `python -m src.engine.evaluate --config <workflow.yaml>`. Hyperparameters,
> model paths, metrics, data CSVs, and output paths should be set in YAML or in
> `src/config/defaults.py`, not passed as long CLI argument lists.

> **Auto-derived paths:** `resolve_config` creates a dataset-first layout under
> `/data/fzliang/reid-project/<dataset>/`:
> - `extract` outputs, including reusable skeleton JSONs → `skeleton/<extractor>/`
> - `prepare` sequence NPZs and window CSVs → `preprocessed/<project>/`
> - `train` outputs → `artifacts/<project>/train/`
> - `evaluate` outputs → `artifacts/<project>/evaluate/`
> - `paths.data_root`, `train_csv`, `val_csv`, `test_csv` are inferred automatically.
> You only need to override them for non-standard layouts.

> **Config fragments:** per-component defaults are loaded automatically from
> `configs/trackers/{name}.yaml` and `configs/pose_estimators/{name}.yaml`.
> You only need to list keys in the workflow YAML when you want to override them.

```bash
./run.sh configs/totalcapture_video_test.yaml all
```

The pipeline will:
1. **Extract** — optionally run/import skeleton extraction into `<dataset>/skeleton/<extractor>`
2. **Prepare** — scan raw data, generate unified `npz`, then write window CSVs;
   for extracted skeletons, first align them into
   derived NPZs via bbox IoU
3. **Train** — train the IMU-Video matcher in independent-person mode
4. **Evaluate** — run the two official metrics: **FrameAcc** and **Group Test**

---

## Official Evaluation

The pipeline `evaluate` stage routes all datasets through one evaluator:

```bash
python -m src.engine.evaluate --config configs/custom.yaml
```

Only two official metrics are exposed:

1. **FrameAcc** — per-window synchronous assignment accuracy. Rows sharing the
   same sequence/session/window form one candidate set; Hungarian matching is
   run on the IMU-video similarity matrix.
2. **Group Test** — sampled group matching over sequence chunks. This reports
   retrieval-style accuracy for group sizes such as 2, 4, 6, 8, and 16.

Historical scripts under `experiments/` are archived research records and are
not part of the official reproduction path. New code should use
`src.engine.evaluate`.

Recommended YACS-style config shape:

```yaml
PROJECT: custom_complete

PREPROCESS:
  DATASET: custom
  RAW_ROOT: /data/fzliang/custom

SLICE:
  TRAIN_SESSIONS: "20260211_171423,20260211_171724,20260211_172522"
  VAL_SESSIONS: "20260211_172257"
  TEST_SESSIONS: "20260211_172257"

TRAIN:
  OUTPUT:
    RUN_NAME: custom_complete_without_filter_20
  DEVICE: cuda:0
  IMU_SENSOR: ""        # keep raw 7D IMU for the hybrid encoder
  REPEAT_SINGLE_SENSOR: 1
  MODEL:
    TYPE: hybrid
    HYBRID_HIDDEN: 128
    HYBRID_SKELETON_SMOOTH: 9
    HYBRID_IMU_SMOOTH: 5

TEST:
  OUTPUT:
    RUN_NAME: custom_complete_without_filter_20
  METRICS:
    FRAME_ACC:
      ENABLED: true
    GROUP_TEST:
      ENABLED: true
      GROUP_SIZES: "2,4,6,8,16"
```

Existing lowercase YAML files are still accepted by the compatibility
normalizer, but new configs should prefer the uppercase YACS form.

The default encoder is the hybrid shoulder-vector model:

- video side: raw shoulder-relative pose + shoulder-local arm-vector tokens;
- IMU side: raw 7D LeftWrist acceleration + quaternion;
- hidden size 128, skeleton smoothing kernel 9, IMU smoothing kernel 5.

When `TRAIN.MODEL.TYPE: hybrid`, the config resolver keeps the IMU path in raw
7D mode by setting `TRAIN.IMU_SENSOR: ""` and `TRAIN.REPEAT_SINGLE_SENSOR: 1`.
This prevents legacy single-sensor-to-48D expansion from silently feeding the
hybrid encoder the wrong representation.

The production codebase intentionally exposes only the hybrid encoder pair.
Older encoder implementations belong in experiment archives, not the official
training/evaluation path.

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
  --input_npz /data/fzliang/reid-project/custom/preprocessed/custom_complete/sequences/custom_20260211_171423.npz \
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
  --alphapose_json /data/fzliang/reid-project/custom/skeleton/alphapose/20260211_171423/alphapose_raw/alphapose-results.json \
  --output visual/pred.mp4
```

**Side-by-side GT vs Prediction comparison:**
```bash
python -m src.visual.visualize_bboxes \
  --video /data/fzliang/custom/2person/20260211_171423/video/20260211_171423.mp4 \
  --anno_csv /data/fzliang/custom/2person/annotations/20260211_171423.anno.csv \
  --alphapose_json /data/fzliang/reid-project/custom/skeleton/alphapose/20260211_171423/alphapose_raw/alphapose-results.json \
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
- Use `configs/egohumans_test.yaml` to validate EgoHumans preprocess/slice/train wiring.
- Use `configs/totalcapture_vicon.yaml` or `configs/totalcapture_video.yaml` for full training (~2 hours).
- Video extraction is slow (~5-10 min per video) but only needed once; use `skip_existing: true` to skip already-extracted videos within the same run.
