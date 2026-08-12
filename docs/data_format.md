# Standard Data Format

The official training and evaluation path consumes standardized per-sequence
NPZ files plus window CSV indexes.

## Window CSV

Each row describes one aligned IMU/skeleton window.

Required fields:

- `npz_path`: path to the sequence NPZ, relative to `PATHS.DATA_ROOT`.
- `window_start`: inclusive frame index.
- `window_end`: exclusive frame index.

Common optional fields:

- `subject`: subject/person id used by subject-level training losses.
- `session`: recording/session id.
- `split`: `train`, `val`, or `test`.
- `domain`: dataset/domain label for domain-adversarial training.
- `skeleton_source`: `gt` or `extract`.
- `person_idx`: skeleton person index in multi-person arrays.
- `imu_idx`: IMU person index in multi-person arrays.
- `source_sequence`: original sequence id used for window grouping.
- `source_window_start`: original window start used for window grouping.
- `candidate_group_id`: explicit synchronized or deterministic cross-sequence
  candidate group used by FrameAcc.
- `candidate_index`: stable candidate position within that group.
- `imu_npz_path`: optional IMU source NPZ override, relative to `PATHS.DATA_ROOT`.
- `imu_window_start`: optional IMU source start override.
- `imu_window_end`: optional IMU source end override.

`WindowAlignmentDataset` builds `group_key` from
`source_sequence/session + source_window_start/window_start + window_end`.
When `imu_npz_path`/`imu_window_start`/`imu_window_end` are absent, IMU and
skeleton are read from the same row. Benchmark controls can use these override
fields to express cross-row IMU/video pairing without changing the NPZ schema.

## Sequence NPZ

Required arrays depend on `skeleton_source`.

Always required:

- `imu`: IMU sequence. Supported shapes are `[T, D]` and `[T, N, D]`.

For `skeleton_source=gt`, one of:

- `gt_skeleton`: `[T, N, 17, C]`
- `skeleton`: `[T, 17, C]` compatibility layout

For `skeleton_source=extract`:

- `extract_skeleton`: `[T, N_extract, 17, C]`
- `gt_to_extract_map`: `[T, N_gt]`, with `-1` for missing extracted persons

Optional arrays:

- `gt_skeleton_meters`: root trajectory source for global motion features.
- `gt_bboxes` or `extract_bboxes`: bounding-box trajectory source.
- `schema_version`, `dataset`, `sequence_id`, `frame_ids`, visibility and
  person-id/mapping arrays used by canonical validation and segment FrameAcc.

## IMU Convention

The official hybrid encoder consumes the first 7 IMU channels as
`acc3 + quat4`. Legacy 48D IMU layouts are still supported through explicit
compatibility options, but new preprocessing code should emit 7D IMU streams.
The quaternion order is `w,x,y,z`; sensor location and coordinate frame must be
declared by the dataset protocol rather than inferred from tensor width.
