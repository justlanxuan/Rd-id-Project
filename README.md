# Re-id Project

Config-driven IMU–skeleton alignment and person re-identification for
TotalCapture, EgoHumans, and Custom data.

The public workflow has one entrypoint and three stages:

```text
preprocess -> train -> test
```

Skeleton extraction is an internal part of `preprocess` when an `extract:`
section is enabled. It is not a separate top-level command.

## Quick start

Create the environment and expose the repository packages:

```bash
conda env create -f environment.yml
conda activate reid_project
export PYTHONPATH="$PWD:$PWD/src"
```

Run the complete configured workflow:

```bash
./run_pipeline.py --config configs/totalcapture_vicon_test.yaml
```

The direct executable form uses the active environment's `python3`; activate
`reid_project` first. `python run_pipeline.py ...` remains equivalent.

Run stages independently or in an explicit order:

```bash
./run_pipeline.py --config CONFIG.yaml --stages preprocess
./run_pipeline.py --config CONFIG.yaml --stages train
./run_pipeline.py --config CONFIG.yaml --stages test
./run_pipeline.py --config CONFIG.yaml --stages preprocess,train,test
```

Only the canonical stage names `preprocess`, `train`, and `test` are accepted.

## Configuration contract

Official YAML files use these top-level domains:

```yaml
project: example_run

paths:
  data_home: /path/to/reid-data

preprocess:
  dataset: totalcapture       # totalcapture | egohumans | custom
  raw_root: /path/to/raw-data

# Optional. If present, extraction runs inside preprocess.
extract:
  pose_estimator: alphapose
  force: false
  reuse_existing: true
  invalid_cache_policy: error

slice:
  window_len: 24
  stride: 16

train:
  model:
    type: hybrid
  epochs: 50
  seed: 0
  device: cuda

test:
  checkpoint: ""
  metrics:
    frame_acc:
      enabled: true
    group_test:
      enabled: false
```

To reuse a previously prepared training cache without repeating expensive
extraction or slicing:

```yaml
preprocess:
  dataset: custom
  reuse_prepared: true
  prepared_root: /path/to/prepared/fold
```

Reuse is accepted only after the split CSVs and referenced NPZ files pass
schema, shape, finite-value, temporal-alignment, and split-leakage checks.
Existing files are never treated as valid merely because their paths exist.

## Architecture

The dependency direction is:

```text
raw dataset
  -> DatasetAdapter
  -> canonical sequences and manifests
  -> optional Extractor
  -> WindowAlignmentDataset
  -> Model / ModelOutput
  -> Metric
  -> RunRecord and aggregate results
```

Each replaceable domain owns its own interface and registry:

| Domain | Public construction point | Implementations |
|---|---|---|
| Dataset adapter | `preprocess.adapters.build_dataset_adapter` | TotalCapture, EgoHumans, Custom |
| Extractor | `src.modules.extractors.build_extractor` | AlphaPose full, ByteTrack+AlphaPose, experimental WHAM |
| Training dataset | `src.datasets.WindowAlignmentDataset` | One canonical window reader shared by all datasets |
| Model | `src.models.build_model` | Hybrid IMU–skeleton matcher |
| Metric | `src.metrics.build_metric` | FrameAcc, Group Test |
| Workflow stage | `src.workflow.build_stage` | preprocess, train, test |

There is intentionally no factory that simultaneously owns datasets,
extractors, models, and stages.

### Dataset adapters

- `TotalCaptureAdapter` aligns Xsens IMU and Vicon/camera skeleton data. The
  G6 source protocol uses `L_LowArm` as 7D `acc3 + quaternion-wxyz`.
- `EgoHumansAdapter` reads realistic per-person IMU and synchronized
  multi-person 2D skeletons. The source protocol uses `LeftWrist` 7D IMU.
- `CustomAdapter` supports validated prepared LOSO caches. Raw Custom output is
  rejected if it contains missing or placeholder skeleton/IMU arrays.

Dataset-specific code ends at the canonical artifact boundary. Training and
evaluation do not branch on raw dataset layout.

### Extractors and cache behavior

All extractors implement dependency checks and return a canonical extraction
artifact. Supported registry names are:

- `alphapose_full`: production backend validated by real
  forced short-video smoke tests;
- `bytetrack_alphapose`: tracking plus AlphaPose, requiring
  the complete external repository and weights;
- `wham`: experimental and rejected from production paths unless explicitly
  enabled;
- `hand4whole_pp`: repository-managed Hand4Whole++ compatibility backend,
  producing canonical 3-D H36M-17 skeletons from existing person tracks.

Relevant cache controls are:

```yaml
extract:
  force: false
  reuse_existing: true
  invalid_cache_policy: error  # error | reextract
```

Cache records distinguish `reused`, `extracted`, and `reextracted`, together
with `adopted_existing` or `verified_current_run` provenance. Backend import
failures and empty skeleton JSON are hard failures.

AlphaPose and ByteTrack repositories may be initialized as submodules:

```bash
git submodule update --init --recursive
```

Machine-specific repository, environment, and checkpoint paths belong in
local YAML files, not source defaults.

### Hand4Whole++ compatibility backend

The repository pins the upstream Hand4Whole++ source as
`third-party/Hand4Whole-plus-plus_RELEASE` and installs the pinned WiLoR and
MMPose source revisions under `third-party/_deps/`.  The setup command also
applies the small WiLoR interface patch needed by the H4W++ hand-control
branch:

```bash
git submodule update --init --recursive
conda env create -f environment-h4wpp.yml
conda activate reid_h4wpp
python tools/setup_h4wpp.py --install
python tools/setup_h4wpp.py --check
```

Model checkpoints and SMPL/SMPL-X/MANO/FLAME assets are not redistributed by
this repository because their upstream licenses and file sizes do not permit
vendoring.  Place licensed assets in the relative layout shown by
`tools/setup_h4wpp.py --check`, or copy an existing H4W++ asset tree with:

```bash
python tools/setup_h4wpp.py --weights-root /path/to/h4wpp-assets
python tools/setup_h4wpp.py --download-public
python tools/setup_h4wpp.py --check
```

The runtime has no hidden dependency on the old external project checkout.
Set the following variables when using the extractor:

```bash
export REID_H4WPP_ROOT="$PWD/third-party/Hand4Whole-plus-plus_RELEASE"
export REID_H4WPP_CHECKPOINT="$PWD/models/hand4whole_plus_plus/snapshot_6.pth"
export REID_H4WPP_PYTHON="$(command -v python)"  # reid_h4wpp, not reid_project
export REID_TRACKS_ROOT=/path/to/alphapose/tracks
```

The H4W++ extractor explicitly assembles the external Python paths in its
subprocess environment and emits the repository's H36M-17 schema; it does not
modify `sys.path` or import from an unrelated local project at runtime.  The
Custom SOTA protocol and reproducible three-train/one-test LOSO commands are
documented in
[`experiments/G13:H4WPP/`](experiments/G13:H4WPP/).

After the Custom prepared cache has been generated, the recorded SOTA can be
reproduced with one repository command:

```bash
python tools/run_h4wpp_loso.py --gpu 0
```

### Model compatibility

Models expose a stable forward contract, `ModelOutput`, and explicit
capabilities. Trainers and evaluators query capabilities for fitted input
statistics, validation behavior, similarity, and segment FrameAcc support;
they do not switch on model names.

New checkpoints contain:

- checkpoint schema version;
- canonical model name;
- model capabilities;
- resolved configuration;
- epoch, model state, optimizer state, and validation statistic.

The model-owned checkpoint adapter migrates supported historical Hybrid keys
and rejects incompatible model identities. A registry entry alone is not
enough to claim a new model is supported: its capabilities, checkpoint round
trip, training step, and requested evaluators must also pass.

### Metrics

FrameAcc always reports raw `correct/total` counts. Window FrameAcc records
candidate-group sizes and singleton rates; the default policy rejects
non-discriminative singleton groups. Custom segment FrameAcc additionally
reports per-clip and per-session counts, macro-session accuracy, and
micro/weighted accuracy.

Group Test is available as a diagnostic metric but is not the primary G6
result.

## Canonical artifacts

Prepared roots contain:

```text
prepared_root/
├── sequences/
│   └── *.npz
├── windows_train.csv
├── windows_val.csv
├── windows_test.csv
└── summary.json              # when generated by the current slicer
```

Sequence NPZ files carry dataset/sequence identity, frame or timestamp
alignment, IMU channels, skeletons, visibility/mapping metadata, and source
provenance when available. Window CSV rows identify the split, source
sequence/person/window, NPZ path, and optional candidate group.

The prepared-cache validator checks the dataset-specific split identity
(`subject` for TotalCapture, `session` for EgoHumans/Custom) and the canonical
`source_sequence` identity.

## Training and evaluation entrypoints

The public pipeline delegates to these modules:

```bash
python -m src.engine.train --config CONFIG.yaml
python -m src.engine.evaluate --config CONFIG.yaml
```

Use them for engine-level debugging. User workflows should normally use
`run_pipeline.py` so stage names and artifact contracts remain consistent.

Training writes `best.pt`, `last.pt`, epoch metrics, final metrics, and fitted
statistics when applicable. Formal evaluation writes raw results plus an
immutable run record containing protocol, Git, config, data-manifest,
checkpoint, and result hashes.

## G6 reproducible benchmark

The active benchmark plan and protocol live under:

```text
experiments/G6:official_refactor_and_three_dataset_benchmark/
```

The frozen matrix contains:

- 6 source train/test runs: 2 sources × 3 seeds;
- 24 source-to-Custom zero-shot evaluations;
- 24 source-to-Custom fine-tune train/evaluations;
- 12 Custom direct LOSO train/evaluations;
- 42 training jobs and 66 evaluation jobs in total.

Seeds are `0`, `42`, and `123`. Custom results are retained per held-out
session and aggregated as both macro-session and micro/weighted FrameAcc.

For diagnostics on the original, unmerged Custom tracker output, use a
full-session evaluation config:

```bash
./run_pipeline.py \
  --config configs/evaluation/custom_full_session_egohumans_pretrained.yaml \
  --stages test
```

This mode treats every raw tracker `idx` as an opaque tracklet. Composite IDs
such as `[1,2]` are not expanded, history is stored independently per ID, a
new ID starts with empty history, and state is reset between complete
sessions. It does not create or evaluate Custom segment files. The result JSON
retains both instantaneous Hungarian predictions and causal historical-vote
predictions. History matching is inference-only: it is configured under
`test.metrics.frame_acc` and never participates in training loss, gradients,
or checkpoint creation.

G6 tooling is intentionally separated from the public pipeline:

```bash
# Inspect the required-cell matrix.
python -m tools.g6.build_matrix

# Generate exact data fingerprints.
python -m tools.g6.build_data_manifests --output-dir OUTPUT

# After explicit human protocol lock and an authorized clean snapshot commit,
# create the immutable protocol hash (which binds that Git commit).
python -m tools.g6.lock_protocol \
  --protocol-document PROTOCOL.md \
  --data-manifest-index DATA_INDEX.json \
  --output PROTOCOL_RECORD.json

# Generate all protocol-bound train/evaluation configs.
python -m tools.g6.build_configs \
  --output-dir CONFIG_DIR \
  --protocol-hash HASH \
  --data-manifest-index DATA_INDEX.json \
  --artifact-root ARTIFACT_ROOT

# Verify the dependency graph without launching work.
python -m tools.g6.run_jobs \
  --index CONFIG_DIR/index.json \
  --protocol-record PROTOCOL_RECORD.json \
  --gpus 0 \
  --max-parallel 1 \
  --log-root LOG_ROOT \
  --state STATE.json \
  --dry-run

# Aggregate only after all 66 validated run records exist.
python -m tools.g6.aggregate_results \
  --records-root RECORDS_ROOT \
  --protocol-record PROTOCOL_RECORD.json \
  --output RESULTS.json \
  --markdown-output RESULTS.md
```

The runner requires an explicit physical GPU list, respects training
dependencies, verifies completed artifacts before skipping them, and refuses
to overwrite partial or corrupt checkpoints/run records.

Controlled protocol variants use explicit benchmark profiles. For example,
the G7 stride-24 ablation uses `--profile stride24` for manifest, protocol, and
resolved-config generation. A profile owns its base configs and prepared roots,
so it cannot silently reuse G6's window CSVs.

## Extending the repository

### Add a dataset

1. Implement `DatasetAdapter` in `preprocess/adapters/`.
2. Emit the canonical sequence/window schema.
3. Register the adapter only in the dataset-adapter registry.
4. Add toy malformed-data tests and one real preprocess smoke test.
5. Document the actual split identity, IMU channels/location/frame, skeleton
   joint order/space, timing, and provenance.

Do not create a dataset-specific training Dataset if the canonical reader can
consume the output.

### Add an extractor

1. Implement `VideoSkeletonExtractor`.
2. Declare capabilities and dependency checks.
3. Produce non-empty canonical skeleton JSON plus provenance.
4. Register it in the extractor registry.
5. Test cache reuse, forced extraction, invalid-cache failure/re-extraction,
   and a real short-video run.

### Add a model

1. Implement the model input/output contract and capabilities.
2. Register its builder in the model registry.
3. Add a model-owned checkpoint adapter if migration is required.
4. Verify forward, loss/backward, save/load, legacy behavior if supported, and
   each advertised evaluator.

### Add a metric

1. Implement `EvaluationMetric` over a stable prediction/embedding bundle.
2. Define exact raw counts, exclusions, aggregation, and degenerate-case
   behavior.
3. Register it in the metric registry.
4. Add hand-verifiable contract tests.

## Validation

Run the repository tests and targeted lint checks:

```bash
python -m pytest -q
python -m ruff check run_pipeline.py preprocess src tools/g6 tests
git diff --check
```

Important integration checks include:

- default, independent, and ordered pipeline stages;
- registry aliases and unknown-key failures;
- malformed/placeholder canonical data rejection;
- FrameAcc singleton and candidate-group behavior;
- model checkpoint round trips and legacy migration;
- prepared-cache leakage/content validation;
- G6 matrix, manifest, protocol, config, scheduler, run-record, and result
  completeness checks.

## Legacy policy

Archived experiment files may still describe older module paths or protocols.
They are historical evidence, not public API. New code must not import deleted
`src.pipeline`, `src.pipelines`, or `src.preprocess` modules. The supported
surfaces are the root `run_pipeline.py`, top-level `preprocess` package, and
the domain packages listed above.
