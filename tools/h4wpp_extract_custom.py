"""Run Hand4Whole++ on tracked custom-video person crops.

The official Hand4Whole++ demo is image-oriented and selects one person with
YOLO.  Custom already has multi-person AlphaPose tracks, so this adapter
reuses those boxes/IDs and only runs the H4W++ model on each tracked crop.
It writes the project's AlphaPose-compatible ``skeleton.json`` with 3-D
H36M-17 joints (root-relative SMPL-X coordinates).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _install_legacy_numpy_aliases() -> None:
    # FLAME/MANO pickle files contain legacy chumpy objects.
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _load_tracks(path: Path) -> dict[int, list[dict[str, Any]]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    frames: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        stem = Path(str(entry.get("image_id", "0.jpg"))).stem
        try:
            frame = int(stem)
        except ValueError:
            continue
        box = np.asarray(entry.get("box", [0, 0, 0, 0]), dtype=np.float32).reshape(-1)
        if box.size < 4 or not np.isfinite(box[:4]).all():
            continue
        x, y, w, h = box[:4].tolist()
        if w <= 1 or h <= 1:
            continue
        raw_idx = entry.get("idx", 0)
        track_id = int(raw_idx[0] if isinstance(raw_idx, list) else raw_idx)
        frames.setdefault(frame, []).append(
            {
                "track_id": track_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(entry.get("score", 1.0) or 1.0),
            }
        )
    return frames


def _h36m17_from_smplx(kpt: np.ndarray) -> np.ndarray:
    """Map H4W++ SMPL-X 137-keypoint output to H36M-17 order."""
    # H4W++ ``smpl_x.kpt`` order (the first 25 entries) is:
    # Pelvis, L/R_Hip, L/R_Knee, L/R_Ankle, Neck, L/R_Shoulder,
    # L/R_Elbow, L/R_Wrist, L_Big_toe, L_Small_toe, L_Heel,
    # R_Big_toe, R_Small_toe, R_Heel, L/R_Ear, L/R_Eye, Nose.
    # This is distinct from both COCO and the SMPL-X articulated-joint order.
    body = np.asarray(kpt, dtype=np.float32)[:25]
    out = np.zeros((17, 3), dtype=np.float32)
    out[0] = body[0]  # pelvis
    out[1] = body[2]
    out[2] = body[4]
    out[3] = body[6]
    out[4] = body[1]
    out[5] = body[3]
    out[6] = body[5]
    thorax = (body[8] + body[9]) / 2.0
    out[7] = (body[0] + thorax) / 2.0
    out[8] = thorax
    out[9] = body[24]  # nose
    out[10] = body[24] + 0.5 * (body[24] - body[7])  # head above neck/nose
    out[11] = body[8]
    out[12] = body[10]
    out[13] = body[12]
    out[14] = body[9]
    out[15] = body[11]
    out[16] = body[13]
    return out


def _build_model(h4w_root: Path, checkpoint: Path, device: str):
    import torch
    import torch.backends.cudnn as cudnn
    from torch.nn.parallel.data_parallel import DataParallel

    os.chdir(h4w_root / "demo")
    from config import cfg  # noqa: I001
    from model import get_model  # noqa: I001

    cfg.set_args(False)
    model = DataParallel(get_model("test")).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["network"], strict=False)
    # Do not call ``model.eval()`` on the whole module: Ultralytics' YOLO
    # object exposes ``train`` as its training API, so recursive nn.Module
    # evaluation would accidentally enter YOLO training setup.  This mirrors
    # the official H4W++ demo and freezes the intended submodules directly.
    for module in model.module.trainable_modules + model.module.eval_modules:
        module.eval()
    cudnn.benchmark = True
    return model, cfg, torch


def _crop_batch(frames: list[np.ndarray], boxes: list[list[float]], cfg, torch):
    import torchvision.transforms as transforms
    from utils.preprocessing import get_patch_img, set_aspect_ratio

    transform = transforms.ToTensor()
    patches = []
    for original_img, box in zip(frames, boxes, strict=True):
        bbox = set_aspect_ratio(box, cfg.input_img_shape[1] / cfg.input_img_shape[0])
        patch, _, _ = get_patch_img(
            original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
        )
        patches.append(transform(patch.astype(np.float32)) / 255.0)
    return torch.stack(patches, dim=0)


def run(args: argparse.Namespace) -> Path:
    _install_legacy_numpy_aliases()
    h4w_root = Path(args.h4w_root).expanduser().resolve()
    video = Path(args.video).expanduser().resolve()
    tracks_path = Path(args.tracks).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tracks = _load_tracks(tracks_path)
    model, cfg, torch = _build_model(
        h4w_root, Path(args.checkpoint).expanduser().resolve(), args.device
    )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    results: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    frame_index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        detections = tracks.get(frame_index, [])
        if args.frame_stride > 1 and frame_index % int(args.frame_stride) != 0:
            detections = []
        if detections:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for start in range(0, len(detections), batch_size):
                chunk = detections[start : start + batch_size]
                imgs = _crop_batch(
                    [rgb] * len(chunk), [item["bbox"] for item in chunk], cfg, torch
                ).to(args.device)
                with torch.no_grad():
                    out = model({"img": imgs}, {}, {}, "test")
                kpts = out["smplx_kpt_cam"].detach().cpu().numpy()
                for item, kpt in zip(chunk, kpts, strict=True):
                    skeleton = _h36m17_from_smplx(kpt)
                    x, y, w, h = item["bbox"]
                    results.append(
                        {
                            "image_id": f"{frame_index}.jpg",
                            "category_id": 1,
                            "keypoints": skeleton.reshape(-1).astype(float).tolist(),
                            "score": float(item["score"]),
                            "box": [x, y, w, h],
                            "idx": int(item["track_id"]),
                        }
                    )
                del imgs, out
        frame_index += 1
    cap.release()
    output.write_text(json.dumps(results), encoding="utf-8")
    print(f"Wrote {len(results)} H4W++ detections across {frame_index} frames to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4w-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
