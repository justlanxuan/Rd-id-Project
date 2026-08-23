# G12 E1 Orientation Inventory

- Schema: `g12.orientation_inventory.v1`
- Read-only: `True`
- Sample limit: `3`; hash limit: `50`

## Summary

| Source | Class | Status | Files | Samples | Failures | Provenance |
|---|---|---|---:|---:|---:|---|
| `totalcapture_vicon_orientation` | `direct` | `candidate` | 1 raw + 10 archives (92 members) | 1 | 0 | TotalCapture Vicon optical motion capture; raw orientation files are not currently passed by the canonical adapter |
| `totalcapture_smplx_root_orientation` | `direct` | `candidate` | 37 | 3 | 0 | TotalCapture SMPL-X processed artifact; root_orient is axis-angle |
| `egohumans_fitted_smpl_global_orientation` | `direct` | `candidate_estimated` | 70113 | 3 | 0 | EgoHumans fitted SMPL estimate; not optical ground truth |
| `egohumans_extracted_pose3d` | `derived` | `derived_only` | 456 | 3 | 0 | EgoHumans 3D pose; heading requires pelvis/hip/shoulder derivation and sign validation |
| `egohumans_fit_pose3d` | `derived` | `derived_only` | 70113 | 3 | 0 | EgoHumans 3D pose; heading requires pelvis/hip/shoulder derivation and sign validation |
| `egohumans_refine_pose3d` | `derived` | `derived_only` | 70113 | 3 | 0 | EgoHumans 3D pose; heading requires pelvis/hip/shoulder derivation and sign validation |
| `fzliang_totalcapture_canonical` | `derived` | `derived_only` | 138 | 3 | 0 | 3D gt_skeleton_meters permits derived heading, but raw Vicon orientation is not present in canonical files |
| `fzliang_egohumans_canonical` | `proxy` | `orientation_missing` | 114192 | 3 | 0 | gt_skeleton is 2D xy plus visibility; no reliable world orientation |
| `fzliang_custom_canonical` | `missing` | `orientation_missing` | 2209 | 3 | 0 | skeleton/extract_skeleton is 2D; current corrected cache is orientation-missing for world yaw |

## Source details

### `totalcapture_vicon_orientation`

- Orientation class: `direct`
- Status: `candidate`
- Coordinate frame: global, exact convention pending
- Time fields: implicit row index
- Provenance: TotalCapture Vicon optical motion capture; raw orientation files are not currently passed by the canonical adapter
- Fingerprint: `4ed7955745e5594209ce663d04cb51c78d1e5e74e60ef5d518da49a451f8bc5e` (all_files)

Representative samples:
- `ReID_imu_generation/data/raw/totalcapture/S1_freestyle3/gt_skel_gbl_ori.txt`: `{"coordinate_frame": "global (declared by gbl filename; convention still requires audit)", "file": {"path": "/data/lyxie/ReID_imu_generation/data/raw/totalcapture/S1_freestyle3/gt_skel_gbl_ori.txt", "relative_path": "ReID_imu_generation/data/raw/totalcapture/S1_freestyle3/gt_skel_gbl_ori.txt", "sha256": "91eee4d3a28f1a4ca6f49985efc5c883640cfad5ea75c9adf6241441ca2433b2", "size_bytes": 1615917}, "finite": true, "format": "quaternion_wxyz", "header_joint_count": 21, "header_joints": ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot"], "quaternion_norm_max": 1.0000008701461214, "quaternion_norm_min": 0.9999991169416101, "timestamp_field": "implicit row index; source sampling must be resolved from protocol", "value_shape": [2040, 84]}`

### `totalcapture_smplx_root_orientation`

- Orientation class: `direct`
- Status: `candidate`
- Coordinate frame: SMPL-X global frame, convention pending
- Time fields: implicit mocap frame index, mocap_frame_rate when present
- Provenance: TotalCapture SMPL-X processed artifact; root_orient is axis-angle
- Fingerprint: `ee6ae99f262014b24905e701f7ef3c65d137397405d68a0cd185a2303d5a868a` (all_files)

Representative samples:
- `ReID_imu_generation/data/processed/totalcapture_test/S1_acting1/s1_acting1_smplx.npz`: `{"coordinate_frame": "SMPL-X global frame; exact relation to Vicon pending", "file": {"path": "/data/lyxie/ReID_imu_generation/data/processed/totalcapture_test/S1_acting1/s1_acting1_smplx.npz", "relative_path": "ReID_imu_generation/data/processed/totalcapture_test/S1_acting1/s1_acting1_smplx.npz", "sha256": "5f69522360ccecde31dc9a28f7d012f9c9792a64fd3532ddf3f80a651707f92d", "size_bytes": 10967422}, "keys": ["gender", "surface_model_type", "mocap_frame_rate", "mocap_time_length", "markers_latent", "latent_labels", "markers_latent_vids", "trans", "poses", "betas", "num_betas", "root_orient", "pose_body", "pose_hand", "pose_jaw", "pose_eye"], "mocap_frame_rate": 60.0, "pose_body": {"dtype": "float64", "finite": true, "max": 1.9992816010697845, "min": -2.133156262456672, "shape": [4114, 63], "size": 259182}, "root_orient": {"dtype": "float64", "finite": true, "max": 2.8373444530934524, "min": -4.914546906942997, "shape": [4114, 3], "size": 12342}, "timestamp_field": "implicit mocap frame index"}`
- `ReID_imu_generation/data/processed/totalcapture_test/S2_rom3/s2_rom3_smplx.npz`: `{"coordinate_frame": "SMPL-X global frame; exact relation to Vicon pending", "file": {"path": "/data/lyxie/ReID_imu_generation/data/processed/totalcapture_test/S2_rom3/s2_rom3_smplx.npz", "relative_path": "ReID_imu_generation/data/processed/totalcapture_test/S2_rom3/s2_rom3_smplx.npz", "sha256": "52cc2a2cf299ee0ec8f24778cba5ff8de8f5d8cb4ea4d8bff70feb0114b7daef", "size_bytes": 15975742}, "keys": ["gender", "surface_model_type", "mocap_frame_rate", "mocap_time_length", "markers_latent", "latent_labels", "markers_latent_vids", "trans", "poses", "betas", "num_betas", "root_orient", "pose_body", "pose_hand", "pose_jaw", "pose_eye"], "mocap_frame_rate": 60.0, "pose_body": {"dtype": "float64", "finite": true, "max": 2.479988784118127, "min": -2.3933316063990335, "shape": [5994, 63], "size": 377622}, "root_orient": {"dtype": "float64", "finite": true, "max": 2.5735234673286125, "min": -1.071087607346821, "shape": [5994, 3], "size": 17982}, "timestamp_field": "implicit mocap frame index"}`
- `ReID_imu_generation/data/processed/totalcapture_test/S5_walking2/s5_walking2_smplx.npz`: `{"coordinate_frame": "SMPL-X global frame; exact relation to Vicon pending", "file": {"path": "/data/lyxie/ReID_imu_generation/data/processed/totalcapture_test/S5_walking2/s5_walking2_smplx.npz", "relative_path": "ReID_imu_generation/data/processed/totalcapture_test/S5_walking2/s5_walking2_smplx.npz", "sha256": "7d1bdfa02edbcac28a0867a35d1fb7a9c7d4da3b59066d5fc401bdc4d0aacaa2", "size_bytes": 9963094}, "keys": ["gender", "surface_model_type", "mocap_frame_rate", "mocap_time_length", "markers_latent", "latent_labels", "markers_latent_vids", "trans", "poses", "betas", "num_betas", "root_orient", "pose_body", "pose_hand", "pose_jaw", "pose_eye"], "mocap_frame_rate": 60.0, "pose_body": {"dtype": "float64", "finite": true, "max": 1.9074449269681286, "min": -1.4975034691899414, "shape": [3737, 63], "size": 235431}, "root_orient": {"dtype": "float64", "finite": true, "max": 2.8303950950683867, "min": -4.866278520746924, "shape": [3737, 3], "size": 11211}, "timestamp_field": "implicit mocap frame index"}`

### `egohumans_fitted_smpl_global_orientation`

- Orientation class: `direct`
- Status: `candidate_estimated`
- Coordinate frame: SMPL fitted global frame, convention pending
- Time fields: filename frame index
- Provenance: EgoHumans fitted SMPL estimate; not optical ground truth
- Fingerprint: `92a0bba642336bb67ea3e683afc94753d2215eb2587357d7d18762a2ff85ca3b` (first_middle_last_samples)

Representative samples:
- `ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/smpl/00001.npy`: `{"available_keys": ["betas", "body_pose", "epoch_loss", "global_orient", "joints", "transl", "vertices"], "coordinate_frame": "SMPL fitted global frame; camera/world convention pending", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/smpl/00001.npy", "relative_path": "ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/smpl/00001.npy", "sha256": "5e379a12c728b09076017310fb0fd077aab17c65f0d1c08233c287931bee682f", "size_bytes": 335745}, "global_orient": {"dtype": "float32", "finite": true, "max": 1.2183336019515991, "min": -0.850002110004425, "shape": [3], "size": 3}, "person_count": 4, "person_ids": ["aria01", "aria02", "aria03", "aria04"], "timestamp_field": "filename frame index"}`
- `ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/smpl/00248.npy`: `{"available_keys": ["betas", "body_pose", "epoch_loss", "global_orient", "joints", "transl", "vertices"], "coordinate_frame": "SMPL fitted global frame; camera/world convention pending", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/smpl/00248.npy", "relative_path": "ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/smpl/00248.npy", "sha256": "6db2724cd459b788d793378a22acb296c44858968025817dc789603979717751", "size_bytes": 335745}, "global_orient": {"dtype": "float32", "finite": true, "max": 1.5544655323028564, "min": 0.03381207585334778, "shape": [3], "size": 3}, "person_count": 4, "person_ids": ["aria01", "aria02", "aria03", "aria04"], "timestamp_field": "filename frame index"}`
- `ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/smpl/00901.npy`: `{"available_keys": ["betas", "body_pose", "epoch_loss", "global_orient", "joints", "transl", "vertices"], "coordinate_frame": "SMPL fitted global frame; camera/world convention pending", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/smpl/00901.npy", "relative_path": "ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/smpl/00901.npy", "sha256": "3fe2944351ba0451f6fa92db431da72a0ad50af3aa88440154a6706207086a03", "size_bytes": 168083}, "global_orient": {"dtype": "float32", "finite": true, "max": 3.0117874145507812, "min": -1.9283478260040283, "shape": [3], "size": 3}, "person_count": 2, "person_ids": ["aria01", "aria02"], "timestamp_field": "filename frame index"}`

### `egohumans_extracted_pose3d`

- Orientation class: `derived`
- Status: `derived_only`
- Coordinate frame: unknown 3D pose frame
- Time fields: implicit row/frame index
- Provenance: EgoHumans 3D pose; heading requires pelvis/hip/shoulder derivation and sign validation
- Fingerprint: `e6129bde4df33f372592851db3a8e7e2add7385bea2bee39ba88ae4724d890d0` (first_middle_last_samples)

Representative samples:
- `ReID/Data/egohumans/extracted_data/01_001_aria01.npy`: `{"array": {"dtype": "float32", "finite": true, "max": 4.584251880645752, "min": -3.481520175933838, "shape": [601, 24, 3], "size": 43272}, "coordinate_frame": "3D pose frame pending convention audit", "field": "pose3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/extracted_data/01_001_aria01.npy", "relative_path": "ReID/Data/egohumans/extracted_data/01_001_aria01.npy", "sha256": "55849948f1583d642d33d0218fbaa6fee562776f766eac30f58e130d0848ff52", "size_bytes": 339461}, "people": null, "timestamp_field": "implicit row/frame index"}`
- `ReID/Data/egohumans/extracted_data/06_011_aria02.npy`: `{"array": {"dtype": "float32", "finite": true, "max": 3.4200057983398438, "min": -1.6257469654083252, "shape": [201, 24, 3], "size": 14472}, "coordinate_frame": "3D pose frame pending convention audit", "field": "pose3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/extracted_data/06_011_aria02.npy", "relative_path": "ReID/Data/egohumans/extracted_data/06_011_aria02.npy", "sha256": "1b01b30e1f45b39c917e87d970fac213532d6228faddabfbcc8e2bd8c58f07f3", "size_bytes": 113858}, "people": null, "timestamp_field": "implicit row/frame index"}`
- `ReID/Data/egohumans/extracted_data/07_013_aria02.npy`: `{"array": {"dtype": "float32", "finite": true, "max": 10.066949844360352, "min": -1.6882450580596924, "shape": [901, 24, 3], "size": 64872}, "coordinate_frame": "3D pose frame pending convention audit", "field": "pose3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/extracted_data/07_013_aria02.npy", "relative_path": "ReID/Data/egohumans/extracted_data/07_013_aria02.npy", "sha256": "0912ea9d9a3a9f1e78d5e01890afbbdfb749af3670c318d7673b875bcbd8a3ab", "size_bytes": 631238}, "people": null, "timestamp_field": "implicit row/frame index"}`

### `egohumans_fit_pose3d`

- Orientation class: `derived`
- Status: `derived_only`
- Coordinate frame: unknown 3D pose frame
- Time fields: implicit row/frame index
- Provenance: EgoHumans 3D pose; heading requires pelvis/hip/shoulder derivation and sign validation
- Fingerprint: `a7e55645e767859a63235644412a80c0dda21b5349a39c32284233109e001b54` (first_middle_last_samples)

Representative samples:
- `ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/fit_poses3d/00001.npy`: `{"array": {"dtype": "float64", "finite": true, "max": 1.0, "min": -3.5700512143632945, "shape": [17, 4], "size": 68}, "coordinate_frame": "3D pose frame pending convention audit", "field": "fit_poses3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/fit_poses3d/00001.npy", "relative_path": "ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/fit_poses3d/00001.npy", "sha256": "76596b314fd9d1572265c264ed9eb67de476749397794f1dcecf97de9538b6c4", "size_bytes": 2721}, "people": ["aria01", "aria02", "aria03", "aria04"], "timestamp_field": "implicit row/frame index"}`
- `ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/fit_poses3d/00248.npy`: `{"array": {"dtype": "float64", "finite": true, "max": 1.0, "min": -1.4766862038793394, "shape": [17, 4], "size": 68}, "coordinate_frame": "3D pose frame pending convention audit", "field": "fit_poses3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/fit_poses3d/00248.npy", "relative_path": "ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/fit_poses3d/00248.npy", "sha256": "89e30e71bf3ee1daa9d0834d9f36333a6bccf4375d51cf95075be5e50144acc8", "size_bytes": 2721}, "people": ["aria01", "aria02", "aria03", "aria04"], "timestamp_field": "implicit row/frame index"}`
- `ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/fit_poses3d/00901.npy`: `{"array": {"dtype": "float64", "finite": true, "max": 13.359232795134844, "min": -9.322373820713562, "shape": [17, 4], "size": 68}, "coordinate_frame": "3D pose frame pending convention audit", "field": "fit_poses3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/fit_poses3d/00901.npy", "relative_path": "ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/fit_poses3d/00901.npy", "sha256": "b0c42e345d4eced88222c184408c5f430cfd61f91bb5098b1080ab539fbd576c", "size_bytes": 1525}, "people": ["aria01", "aria02"], "timestamp_field": "implicit row/frame index"}`

### `egohumans_refine_pose3d`

- Orientation class: `derived`
- Status: `derived_only`
- Coordinate frame: unknown 3D pose frame
- Time fields: implicit row/frame index
- Provenance: EgoHumans 3D pose; heading requires pelvis/hip/shoulder derivation and sign validation
- Fingerprint: `ec85d3d086f57cdcfbeced8a2784164c9fd7533ec9d742c85f9fdad44572534e` (first_middle_last_samples)

Representative samples:
- `ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/refine_poses3d/00001.npy`: `{"array": {"dtype": "float64", "finite": true, "max": 1.0, "min": -3.569370876473835, "shape": [17, 4], "size": 68}, "coordinate_frame": "3D pose frame pending convention audit", "field": "refine_poses3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/refine_poses3d/00001.npy", "relative_path": "ReID/Data/egohumans/data/01_tagging/001_tagging/processed_data/refine_poses3d/00001.npy", "sha256": "639280a9e928cb25e73b8973b322499430425d5750f14246abb4c703fa1d38f5", "size_bytes": 2721}, "people": ["aria01", "aria02", "aria03", "aria04"], "timestamp_field": "implicit row/frame index"}`
- `ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/refine_poses3d/00248.npy`: `{"array": {"dtype": "float64", "finite": true, "max": 1.0, "min": -1.4627470627165517, "shape": [17, 4], "size": 68}, "coordinate_frame": "3D pose frame pending convention audit", "field": "refine_poses3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/refine_poses3d/00248.npy", "relative_path": "ReID/Data/egohumans/data/06_badminton/016_badminton/processed_data/refine_poses3d/00248.npy", "sha256": "2eaf1022c9b2581e39945e9960794a3f9df30a01dc9519ba698815972b83a18e", "size_bytes": 2721}, "people": ["aria01", "aria02", "aria03", "aria04"], "timestamp_field": "implicit row/frame index"}`
- `ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/refine_poses3d/00901.npy`: `{"array": {"dtype": "float64", "finite": true, "max": 13.359486705725375, "min": -9.321067965852551, "shape": [17, 4], "size": 68}, "coordinate_frame": "3D pose frame pending convention audit", "field": "refine_poses3d", "file": {"path": "/data/lyxie/ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/refine_poses3d/00901.npy", "relative_path": "ReID/Data/egohumans/data/07_tennis/013_tennis/processed_data/refine_poses3d/00901.npy", "sha256": "088160bfa055ecdb1f32b0ce403a8e83d9073f2776e48322852f902fcd83a3d6", "size_bytes": 1525}, "people": ["aria01", "aria02"], "timestamp_field": "implicit row/frame index"}`

### `fzliang_totalcapture_canonical`

- Orientation class: `derived`
- Status: `derived_only`
- Coordinate frame: canonical frame, pending source-specific audit
- Time fields: timestamps_s when present, frame_ids
- Provenance: 3D gt_skeleton_meters permits derived heading, but raw Vicon orientation is not present in canonical files
- Fingerprint: `2cfedf22ad09d4679836197f0d194fc5fe2e0452fbfef7645822409452211057` (first_middle_last_samples)

Representative samples:
- `reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences/totalcapture_S1_acting1_cam1.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "totalcapture", "file": {"path": "/data/fzliang/reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences/totalcapture_S1_acting1_cam1.npz", "relative_path": "reid-project/totalcapture/preprocessed/g6_totalcapture_source/sequences/totalcapture_S1_acting1_cam1.npz", "sha256": "b42f6c4a6a86199b1fd21627299680ca1262056a73fd1d7bacaa4705ed898fb3", "size_bytes": 1604761}, "keys": ["schema_version", "video_path", "dataset", "sequence_id", "frame_ids", "imu", "imu_channels", "imu_location", "imu_ids", "gt_person_ids", "gt_bboxes", "gt_visibility", "gt_skeleton", "gt_skeleton_meters"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 125.6760025024414, "min": -123.04100036621094, "shape": [4115, 1, 17, 3], "size": 209865}, "skeleton_field": "gt_skeleton_meters", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`
- `reid-project/totalcapture/preprocessed/stride24_w24/sequences/totalcapture_S2_walking2_cam1.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "totalcapture", "file": {"path": "/data/fzliang/reid-project/totalcapture/preprocessed/stride24_w24/sequences/totalcapture_S2_walking2_cam1.npz", "relative_path": "reid-project/totalcapture/preprocessed/stride24_w24/sequences/totalcapture_S2_walking2_cam1.npz", "sha256": "9c4d87d4d78015845a8a854e7a8674d3ecc021b3f5ea772693a513e95d4d494d", "size_bytes": 1402717}, "keys": ["schema_version", "video_path", "dataset", "sequence_id", "frame_ids", "imu", "imu_channels", "imu_location", "imu_ids", "gt_person_ids", "gt_bboxes", "gt_visibility", "gt_skeleton", "gt_skeleton_meters"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 91.47969818115234, "min": -103.02200317382812, "shape": [3575, 1, 17, 3], "size": 182325}, "skeleton_field": "gt_skeleton_meters", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`
- `reid-project/totalcapture/preprocessed/totalcapture_test/sequences/totalcapture_S5_walking2_cam1.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "totalcapture", "file": {"path": "/data/fzliang/reid-project/totalcapture/preprocessed/totalcapture_test/sequences/totalcapture_S5_walking2_cam1.npz", "relative_path": "reid-project/totalcapture/preprocessed/totalcapture_test/sequences/totalcapture_S5_walking2_cam1.npz", "sha256": "17ed55c784a802153ff770f6eb6a1f05ef86912bd72dcc3bad50c307bbe76731", "size_bytes": 2042052}, "keys": ["video_path", "dataset", "sequence_id", "frame_ids", "imu", "imu_ids", "gt_person_ids", "gt_bboxes", "gt_visibility", "gt_skeleton", "gt_skeleton_meters"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 114.20500183105469, "min": -134.19400024414062, "shape": [3738, 1, 17, 3], "size": 190638}, "skeleton_field": "gt_skeleton_meters", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`

### `fzliang_egohumans_canonical`

- Orientation class: `proxy`
- Status: `orientation_missing`
- Coordinate frame: canonical frame, pending source-specific audit
- Time fields: timestamps_s when present, frame_ids
- Provenance: gt_skeleton is 2D xy plus visibility; no reliable world orientation
- Fingerprint: `83ccab11926fe9b3946bfd844ea95f2b3953a93bd2ac19342a77fac43e8101e4` (first_middle_last_samples)

Representative samples:
- `reid-project/egohumans/preprocessed/e21_source_cache_compat/sequences/e21src_test_01_001_0_100.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "egohumans", "file": {"path": "/data/fzliang/reid-project/egohumans/preprocessed/e21_source_cache_compat/sequences/e21src_test_01_001_0_100.npz", "relative_path": "reid-project/egohumans/preprocessed/e21_source_cache_compat/sequences/e21src_test_01_001_0_100.npz", "sha256": "9cfdb55730cea4273f70c5029702ab45c3dc7ddb2ca91e90bb0ea9fb7dd708df", "size_bytes": 53666}, "keys": ["gt_skeleton", "imu"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 3352.06640625, "min": 0.0, "shape": [100, 4, 2, 17], "size": 13600}, "skeleton_field": "gt_skeleton", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`
- `reid-project/egohumans/preprocessed/e28_aug_a14p25_lightnoise_w24/sequences/e28_a14_p25__noise_light_acc0p05_gyro0p02_mount_yaw_p30_02639_16.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "egohumans", "file": {"path": "/data/fzliang/reid-project/egohumans/preprocessed/e28_aug_a14p25_lightnoise_w24/sequences/e28_a14_p25__noise_light_acc0p05_gyro0p02_mount_yaw_p30_02639_16.npz", "relative_path": "reid-project/egohumans/preprocessed/e28_aug_a14p25_lightnoise_w24/sequences/e28_a14_p25__noise_light_acc0p05_gyro0p02_mount_yaw_p30_02639_16.npz", "sha256": "374ff89ecd3a0b06dd76bd1a4e230a222187fa630cb7609af541ea8cc7755875", "size_bytes": 3881}, "keys": ["skeleton", "imu"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 1636.845947265625, "min": 502.76385498046875, "shape": [24, 17, 2], "size": 816}, "skeleton_field": "skeleton", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`
- `reid-project/egohumans/preprocessed/stride24_w24/sequences/egohumans_03_010.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "egohumans", "file": {"path": "/data/fzliang/reid-project/egohumans/preprocessed/stride24_w24/sequences/egohumans_03_010.npz", "relative_path": "reid-project/egohumans/preprocessed/stride24_w24/sequences/egohumans_03_010.npz", "sha256": "5d022fdc2b04ca842d19cde2e376d928cf8b317cf002fc2d9517644b61895b04", "size_bytes": 257855}, "keys": ["schema_version", "video_path", "dataset", "sequence_id", "frame_ids", "imu", "imu_channels", "imu_location", "imu_ids", "gt_person_ids", "gt_bboxes", "gt_visibility", "gt_skeleton"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 3150.548583984375, "min": 0.0, "shape": [551, 3, 17, 3], "size": 84303}, "skeleton_field": "gt_skeleton", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`

### `fzliang_custom_canonical`

- Orientation class: `missing`
- Status: `orientation_missing`
- Coordinate frame: canonical frame, pending source-specific audit
- Time fields: timestamps_s when present, frame_ids
- Provenance: skeleton/extract_skeleton is 2D; current corrected cache is orientation-missing for world yaw
- Fingerprint: `522e8360aaa94de044698eb3742676b01191a871a71c80f8c1053155ba743639` (first_middle_last_samples)

Representative samples:
- `reid-project/custom/preprocessed/custom_hybrid_finetune_from_egohumans/sequences/custom_20260211_171423.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "custom", "file": {"path": "/data/fzliang/reid-project/custom/preprocessed/custom_hybrid_finetune_from_egohumans/sequences/custom_20260211_171423.npz", "relative_path": "reid-project/custom/preprocessed/custom_hybrid_finetune_from_egohumans/sequences/custom_20260211_171423.npz", "sha256": "49bda9765a531acbd3dd5875f69f55a2b24c5e01be438857b320f81612b516c7", "size_bytes": 164127}, "keys": ["video_path", "dataset", "sequence_id", "frame_ids", "imu", "imu_ids", "gt_person_ids", "gt_bboxes", "gt_visibility", "imu_person_map"], "orientation_keys": [], "skeleton": {"dtype": "float64", "finite": true, "shape": [0], "size": 0}, "skeleton_field": "skeleton", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`
- `reid-project/custom/preprocessed/hybrid_w24_fold4_stride8/sequences/custom_20260211_171724_seg1_p0_536_560.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "custom", "file": {"path": "/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_fold4_stride8/sequences/custom_20260211_171724_seg1_p0_536_560.npz", "relative_path": "reid-project/custom/preprocessed/hybrid_w24_fold4_stride8/sequences/custom_20260211_171724_seg1_p0_536_560.npz", "sha256": "89065dfe8b528d19f27264423d6dae102efad8253d5759c16e5bd199b25a993b", "size_bytes": 3714}, "keys": ["skeleton", "imu", "sequence_id", "person_idx", "window_start", "window_end"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 618.8154296875, "min": 115.66632843017578, "shape": [24, 17, 2], "size": 816}, "skeleton_field": "skeleton", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`
- `reid-project/custom/preprocessed/hybrid_w24_fold4_stride8/sequences/custom_20260211_172522_seg1_p1_96_120.npz`: `{"coordinate_frame": "canonical frame; source semantics inherited and must be resolved", "dataset": "custom", "file": {"path": "/data/fzliang/reid-project/custom/preprocessed/hybrid_w24_fold4_stride8/sequences/custom_20260211_172522_seg1_p1_96_120.npz", "relative_path": "reid-project/custom/preprocessed/hybrid_w24_fold4_stride8/sequences/custom_20260211_172522_seg1_p1_96_120.npz", "sha256": "8a7f1cd89c72149eb372ceaddb78aaefb6f6e74fb1e860d5f00ade374c1d9dd8", "size_bytes": 3721}, "keys": ["skeleton", "imu", "sequence_id", "person_idx", "window_start", "window_end"], "orientation_keys": [], "skeleton": {"dtype": "float32", "finite": true, "max": 487.0242614746094, "min": 10.146764755249023, "shape": [24, 17, 2], "size": 816}, "skeleton_field": "skeleton", "timestamp_field": "timestamps_s when present, otherwise frame_ids"}`
