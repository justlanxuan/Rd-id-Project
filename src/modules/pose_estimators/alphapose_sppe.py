"""AlphaPose SPPE adapter (decoupled pose-only estimator).

Supports both in-process single-frame estimation (BasePoseEstimator) and
subprocess detfile mode for ComposedExtractor.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.registry import POSE_ESTIMATORS
from src.preprocess.structures import Pose, Track
from src.modules.pose_estimators.base import BasePoseEstimator


@dataclass
class AlphaPoseSPPEConfig:
    """Configuration for AlphaPose SPPE (pose-only) adapter."""

    repo_root: str = "/home/fzliang/origin/AlphaPose"
    cfg_file: Optional[str] = None
    checkpoint_file: Optional[str] = None
    device: str = "cuda:0"
    pose_batch_size: int = 32
    use_flip: bool = False
    image_is_bgr: bool = False
    expected_commit: Optional[str] = None
    strict_commit: bool = False
    # Subprocess overrides
    python: str = sys.executable
    detbatch: Optional[int] = None
    posebatch: Optional[int] = None
    gpu: Optional[int] = None
    headless: bool = True
    use_expandable_segments: bool = False


@POSE_ESTIMATORS.register("alphapose_sppe")
class AlphaPoseSPPE(BasePoseEstimator):
    """Run AlphaPose keypoint inference on externally provided bounding boxes."""

    def __init__(self, config: Optional[AlphaPoseSPPEConfig] = None):
        self.config = config or AlphaPoseSPPEConfig()
        self.repo_path = Path(self.config.repo_root).expanduser().resolve()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"AlphaPose repo not found: {self.repo_path}")

        self._device: Any = None
        self._cfg = None
        self._pose_model = None
        self._pose_dataset = None
        self._transformation = None
        self._heatmap_to_coord = None
        self._flip = None
        self._flip_heatmap = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        self._validate_commit(self.repo_path)
        self._ensure_import_path(self.repo_path)
        self._device = self._resolve_device(self.config.device)
        self._load_runtime()
        self._initialized = True

    @staticmethod
    def _resolve_device(device_name: str):
        import torch

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _validate_commit(self, repo_path: Path) -> None:
        if not self.config.expected_commit:
            return
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if self.config.strict_commit:
                raise RuntimeError(
                    "Failed to read AlphaPose git commit: "
                    f"{result.stderr.strip() or 'unknown error'}"
                )
            return
        actual = result.stdout.strip()
        expected = self.config.expected_commit.strip()
        if actual != expected:
            message = f"AlphaPose commit mismatch: expected {expected}, got {actual}."
            if self.config.strict_commit:
                raise RuntimeError(message)
            print(f"[AlphaPoseSPPE] Warning: {message}")

    @staticmethod
    def _ensure_import_path(repo_path: Path) -> None:
        repo_root = str(repo_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _load_runtime(self) -> None:
        if not self.config.cfg_file:
            raise ValueError("AlphaPoseSPPEConfig.cfg_file is required")
        if not self.config.checkpoint_file:
            raise ValueError("AlphaPoseSPPEConfig.checkpoint_file is required")

        import torch

        from alphapose.models import builder
        from alphapose.utils.config import update_config
        from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
        from alphapose.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord

        self._flip = flip
        self._flip_heatmap = flip_heatmap

        self._cfg = update_config(self.config.cfg_file)
        self._pose_model = builder.build_sppe(self._cfg.MODEL, preset_cfg=self._cfg.DATA_PRESET)

        checkpoint = torch.load(self.config.checkpoint_file, map_location=self._device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self._pose_model.load_state_dict(state_dict, strict=False)
        self._pose_model.to(self._device)
        self._pose_model.eval()

        self._pose_dataset = builder.retrieve_dataset(self._cfg.DATASET.TRAIN)
        self._transformation = self._build_transformation(
            SimpleTransform=SimpleTransform,
            SimpleTransform3DSMPL=SimpleTransform3DSMPL,
        )
        self._heatmap_to_coord = get_func_heatmap_to_coord(self._cfg)

    def _build_transformation(self, SimpleTransform, SimpleTransform3DSMPL):
        import torch

        input_size = self._cfg.DATA_PRESET.IMAGE_SIZE
        output_size = self._cfg.DATA_PRESET.HEATMAP_SIZE
        sigma = self._cfg.DATA_PRESET.SIGMA

        if self._cfg.DATA_PRESET.TYPE == "simple":
            return SimpleTransform(
                self._pose_dataset,
                scale_factor=0,
                input_size=input_size,
                output_size=output_size,
                rot=0,
                sigma=sigma,
                train=False,
                add_dpg=False,
                gpu_device=self._device,
            )

        if self._cfg.DATA_PRESET.TYPE == "simple_smpl":
            from easydict import EasyDict as edict

            dummy_set = edict(
                {
                    "joint_pairs_17": None,
                    "joint_pairs_24": None,
                    "joint_pairs_29": None,
                    "bbox_3d_shape": (2.2, 2.2, 2.2),
                }
            )
            return SimpleTransform3DSMPL(
                dummy_set,
                scale_factor=self._cfg.DATASET.SCALE_FACTOR,
                color_factor=self._cfg.DATASET.COLOR_FACTOR,
                occlusion=self._cfg.DATASET.OCCLUSION,
                input_size=self._cfg.MODEL.IMAGE_SIZE,
                output_size=self._cfg.MODEL.HEATMAP_SIZE,
                depth_dim=self._cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=self._cfg.DATASET.ROT_FACTOR,
                sigma=self._cfg.MODEL.EXTRA.SIGMA,
                train=False,
                add_dpg=False,
                gpu_device=self._device,
                loss_type=self._cfg.LOSS["TYPE"],
            )

        raise ValueError(f"Unsupported AlphaPose preset type: {self._cfg.DATA_PRESET.TYPE}")

    def reset(self) -> None:
        """Reset any sequence-local state."""
        self._initialized = False
        self._pose_model = None
        self._pose_dataset = None
        self._transformation = None
        self._heatmap_to_coord = None

    def estimate(self, frame: np.ndarray, tracks: List[Track]) -> List[Pose]:
        """Estimate poses for tracked bounding boxes in a single frame."""
        self._lazy_init()
        orig_img = self._prepare_image(frame)
        if not tracks:
            return []
        boxes = np.stack([t.detection.bbox for t in tracks], axis=0)
        track_ids = [t.track_id for t in tracks]
        box_scores = [t.detection.score for t in tracks]

        inps, cropped_boxes = self._build_pose_inputs(orig_img, boxes)
        heatmaps = self._forward_pose_model(inps)
        raw_results = self._decode_results(
            orig_img=orig_img,
            boxes=boxes,
            cropped_boxes=cropped_boxes,
            heatmaps=heatmaps,
            track_ids=track_ids,
            box_scores=box_scores,
        )

        poses: List[Pose] = []
        for r in raw_results:
            kpts = np.stack(
                [
                    np.array(r["keypoints_xy"], dtype=np.float32),
                    np.array(r["keypoints_score"], dtype=np.float32),
                ],
                axis=-1,
            )  # not right: keypoints_xy is [K,2], keypoints_score is [K]
            # Let's fix:
            xy = np.asarray(r["keypoints_xy"], dtype=np.float32)  # [K, 2]
            scores = np.asarray(r["keypoints_score"], dtype=np.float32)  # [K]
            keypoints = np.concatenate([xy, scores[:, None]], axis=-1)  # [K, 3]
            poses.append(
                Pose(
                    keypoints=keypoints,
                    bbox=np.asarray(r["bbox_xyxy"], dtype=np.float32),
                    score=float(r["pose_score"]),
                    track_id=r["track_id"],
                )
            )
        return poses

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"image must be HWC, got {arr.shape}")
        if self.config.image_is_bgr:
            return arr[:, :, ::-1].copy()
        return arr.copy()

    def _build_pose_inputs(self, orig_img: np.ndarray, boxes: np.ndarray):
        import torch

        input_size = self._cfg.DATA_PRESET.IMAGE_SIZE
        inps = torch.zeros(boxes.shape[0], 3, *input_size)
        cropped_boxes = torch.zeros(boxes.shape[0], 4)
        for idx, box in enumerate(boxes):
            inps[idx], cropped_box = self._transformation.test_transform(orig_img, box)
            cropped_boxes[idx] = torch.as_tensor(cropped_box, dtype=torch.float32)
        return inps, cropped_boxes

    def _forward_pose_model(self, inps) -> Any:
        import torch

        batch_size = max(1, int(self.config.pose_batch_size))
        if self.config.use_flip:
            batch_size = max(1, batch_size // 2)

        datalen = inps.size(0)
        leftover = 1 if datalen % batch_size else 0
        num_batches = datalen // batch_size + leftover

        heatmap_chunks = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch = inps[batch_idx * batch_size : min((batch_idx + 1) * batch_size, datalen)]
                batch = batch.to(self._device)
                if self.config.use_flip:
                    batch = torch.cat((batch, self._flip(batch)))
                hm = self._pose_model(batch)
                if self.config.use_flip:
                    hm_flip = self._flip_heatmap(
                        hm[int(len(hm) / 2) :],
                        self._pose_dataset.joint_pairs,
                        shift=True,
                    )
                    hm = (hm[0 : int(len(hm) / 2)] + hm_flip) / 2
                heatmap_chunks.append(hm)
        return torch.cat(heatmap_chunks)

    @staticmethod
    def _select_eval_joints(num_joints: int) -> List[int]:
        if num_joints in (136, 133):
            return list(range(num_joints))
        if num_joints == 68:
            return list(range(68))
        if num_joints in (26, 21):
            return list(range(num_joints))
        return list(range(num_joints))

    def _decode_results(
        self,
        orig_img: np.ndarray,
        boxes: np.ndarray,
        cropped_boxes: Any,
        heatmaps: Any,
        track_ids: Sequence[Optional[int]],
        box_scores: Sequence[Optional[float]],
    ) -> List[Dict[str, Any]]:
        norm_type = self._cfg.LOSS.get("NORM_TYPE", None)
        hm_size = self._cfg.DATA_PRESET.HEATMAP_SIZE
        eval_joints = self._select_eval_joints(heatmaps.size(1))

        results: List[Dict[str, Any]] = []
        for idx in range(heatmaps.shape[0]):
            bbox = cropped_boxes[idx].tolist()
            if isinstance(self._heatmap_to_coord, list):
                face_hand_num = 110 if heatmaps.size(1) != 68 else 42
                pose_coords_body_foot, pose_scores_body_foot = self._heatmap_to_coord[0](
                    heatmaps[idx][eval_joints[:-face_hand_num]],
                    bbox,
                    hm_shape=hm_size,
                    norm_type=norm_type,
                )
                pose_coords_face_hand, pose_scores_face_hand = self._heatmap_to_coord[1](
                    heatmaps[idx][eval_joints[-face_hand_num:]],
                    bbox,
                    hm_shape=hm_size,
                    norm_type=norm_type,
                )
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = self._heatmap_to_coord(
                    heatmaps[idx][eval_joints],
                    bbox,
                    hm_shape=hm_size,
                    norm_type=norm_type,
                )

            results.append(
                {
                    "track_id": None if track_ids[idx] is None else int(track_ids[idx]),
                    "bbox_xyxy": [float(v) for v in boxes[idx].tolist()],
                    "bbox_score": None if box_scores[idx] is None else float(box_scores[idx]),
                    "keypoints_xy": pose_coord.tolist() if hasattr(pose_coord, "tolist") else pose_coord,
                    "keypoints_score": pose_score.tolist() if hasattr(pose_score, "tolist") else pose_score,
                    "pose_score": float(np.mean(pose_score)) if len(pose_score) else 0.0,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Subprocess helper for ComposedExtractor
    # ------------------------------------------------------------------
    def run_on_video(
        self,
        video_path: str,
        output_dir: str,
        detfile: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Run AlphaPose CLI in detfile mode and return skeleton.json path."""
        if detfile is None:
            raise ValueError("AlphaPoseSPPE subprocess mode requires a detfile")
        ap_outdir = Path(output_dir) / "alphapose_raw"
        ap_outdir.mkdir(parents=True, exist_ok=True)
        json_path = ap_outdir / "alphapose-results.json"
        skeleton_json = Path(output_dir) / "skeleton.json"

        cfg = self.config.cfg_file
        ckpt = self.config.checkpoint_file
        if not cfg or not ckpt:
            raise ValueError("AlphaPoseSPPEConfig requires cfg_file and checkpoint_file")

        cfg_path = Path(cfg)
        if not cfg_path.is_absolute():
            cfg_path = self.repo_path / cfg_path
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = self.repo_path / ckpt_path

        cmd = [
            self.config.python,
            "scripts/demo_inference.py",
            "--cfg",
            str(cfg_path),
            "--checkpoint",
            str(ckpt_path),
            "--detfile",
            str(detfile),
            "--outdir",
            str(ap_outdir),
        ]
        if self.config.detbatch is not None:
            cmd.extend(["--detbatch", str(self.config.detbatch)])
        if self.config.posebatch is not None:
            cmd.extend(["--posebatch", str(self.config.posebatch)])

        _env = dict(env) if env is not None else os.environ.copy()
        # Remove Autism-project src paths to avoid conflicting with AlphaPose's internal utils package
        filtered_paths = [
            p for p in _env.get("PYTHONPATH", "").split(os.pathsep)
            if p and "Autism-project" not in p
        ]
        alphapose_paths = [
            str(self.repo_path),
            str(self.repo_path / "trackers"),
            str(self.repo_path / "detector" / "tracker"),
        ]
        _env["PYTHONPATH"] = os.pathsep.join(alphapose_paths)
        if filtered_paths:
            _env["PYTHONPATH"] = f"{_env['PYTHONPATH']}{os.pathsep}{os.pathsep.join(filtered_paths)}"
        if self.config.gpu is not None:
            _env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu)
        else:
            _env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.config.use_expandable_segments:
            _env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if self.config.headless:
            _env.setdefault("MPLBACKEND", "Agg")
            _env.setdefault("QT_QPA_PLATFORM", "offscreen")
            _env.setdefault("SDL_VIDEODRIVER", "dummy")
            _env.setdefault("DISPLAY", "")
            _env.setdefault("HEADLESS", "1")

        subprocess.run(cmd, check=True, cwd=str(self.repo_path), env=_env)

        if not json_path.exists():
            raise FileNotFoundError(f"AlphaPose JSON not found: {json_path}")
        shutil.copy2(json_path, skeleton_json)
        return skeleton_json
