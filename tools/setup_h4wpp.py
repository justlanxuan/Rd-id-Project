#!/usr/bin/env python3
"""Install and verify the repository-managed Hand4Whole++ compatibility backend.

The upstream H4W++ repository and its model weights are not redistributed by
this project.  This command pins the upstream source revisions, wires WiLoR
and MMPose into the checked-out H4W++ tree, applies the one small interface
patch required by this adapter, and reports every missing licensed asset.

Typical setup from a fresh clone::

    git submodule update --init --recursive
    python tools/setup_h4wpp.py --install --weights-root /path/to/h4wpp-assets
    python tools/setup_h4wpp.py --check --weights-root /path/to/h4wpp-assets
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
H4W_PATH = REPO_ROOT / "third-party" / "Hand4Whole-plus-plus_RELEASE"
DEPS_ROOT = REPO_ROOT / "third-party" / "_deps"
PATCH_PATH = REPO_ROOT / "third-party" / "patches" / "wilor_img_feat.patch"

WILOR_URL = "https://github.com/rolpotamias/WiLoR.git"
WILOR_COMMIT = "fcb911312a38fa8badd30d9656a167485d61b8f9"
MMPOSE_URL = "https://github.com/open-mmlab/mmpose.git"
MMPOSE_COMMIT = "71ec36ebd63c475ab589afc817868e749a61491f"

PUBLIC_ASSETS = {
    "WiLoR detector": (
        "common/nets/WiLoR/pretrained_models/detector.pt",
        "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
    ),
    "WiLoR checkpoint": (
        "common/nets/WiLoR/pretrained_models/wilor_final.ckpt",
        "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt",
    ),
}

REQUIRED_ASSETS = (
    ("Hand4Whole++ inference checkpoint", "demo/snapshot_6.pth"),
    ("H4W++ YOLO detector", "main/yolo11n.pt"),
    ("WiLoR detector", "common/nets/WiLoR/pretrained_models/detector.pt"),
    ("WiLoR checkpoint", "common/nets/WiLoR/pretrained_models/wilor_final.ckpt"),
    ("WiLoR MANO right model", "common/nets/WiLoR/mano_data/MANO_RIGHT.pkl"),
    ("DWPose checkpoint", "common/nets/mmpose/dw-ll_ucoco.pth"),
    ("SMPL neutral model", "common/utils/human_model_files/smpl/SMPL_NEUTRAL.pkl"),
    ("SMPL male model", "common/utils/human_model_files/smpl/SMPL_MALE.pkl"),
    ("SMPL female model", "common/utils/human_model_files/smpl/SMPL_FEMALE.pkl"),
    ("SMPL-X neutral model", "common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl"),
    ("SMPL-X male model", "common/utils/human_model_files/smplx/SMPLX_MALE.npz"),
    ("SMPL-X female model", "common/utils/human_model_files/smplx/SMPLX_FEMALE.npz"),
    ("FLAME neutral model", "common/utils/human_model_files/flame/FLAME_NEUTRAL.pkl"),
    ("FLAME static embedding", "common/utils/human_model_files/flame/flame_static_embedding.pkl"),
    ("FLAME dynamic embedding", "common/utils/human_model_files/flame/flame_dynamic_embedding.npy"),
    ("MANO-SMPL-X vertex IDs", "common/utils/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl"),
    ("SMPL-X FLAME vertex IDs", "common/utils/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy"),
    ("SMPL-X to J14 regressor", "common/utils/human_model_files/smplx/SMPLX_to_J14.pkl"),
    ("MANO left model", "common/utils/human_model_files/mano/MANO_LEFT.pkl"),
    ("MANO right model", "common/utils/human_model_files/mano/MANO_RIGHT.pkl"),
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("[h4wpp-setup] $", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_repo(path: Path, url: str, commit: str) -> None:
    if not path.exists():
        run(["git", "clone", url, str(path)])
    if git_head(path) != commit:
        run(["git", "fetch", "--tags", "origin"], cwd=path)
        run(["git", "checkout", "--detach", commit], cwd=path)
    if git_head(path) != commit:
        raise RuntimeError(f"Pinned dependency revision not available: {path} {commit}")


def replace_link(path: Path, target: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        if any(path.iterdir()):
            raise RuntimeError(f"Refusing to replace non-empty directory: {path}")
        path.rmdir()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)


def wire_dependencies() -> tuple[Path, Path]:
    wilor = DEPS_ROOT / "WiLoR"
    mmpose = DEPS_ROOT / "mmpose"
    ensure_repo(wilor, WILOR_URL, WILOR_COMMIT)
    ensure_repo(mmpose, MMPOSE_URL, MMPOSE_COMMIT)
    replace_link(H4W_PATH / "common" / "nets" / "WiLoR", wilor)
    replace_link(H4W_PATH / "common" / "nets" / "mmpose", mmpose)
    return wilor, mmpose


def apply_wilor_patch(wilor: Path) -> None:
    if not PATCH_PATH.is_file():
        raise FileNotFoundError(f"Missing repository patch: {PATCH_PATH}")
    check = subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", str(PATCH_PATH)],
        cwd=wilor,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode == 0:
        run(["git", "apply", "--unidiff-zero", str(PATCH_PATH)], cwd=wilor)
        return
    reverse = subprocess.run(
        ["git", "apply", "--unidiff-zero", "--reverse", "--check", str(PATCH_PATH)],
        cwd=wilor,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if reverse.returncode != 0:
        raise RuntimeError("WiLoR compatibility patch cannot be applied cleanly")
    print("[h4wpp-setup] WiLoR compatibility patch already applied")


def copy_assets(source_root: Path) -> None:
    """Copy user-provided licensed assets into the pinned H4W++ tree."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Asset root does not exist: {source_root}")
    for _, relative in REQUIRED_ASSETS:
        source = source_root / relative
        target = H4W_PATH / relative
        if not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"[h4wpp-setup] copied {relative}")


def download_public_assets() -> None:
    import urllib.request

    for label, (relative, url) in PUBLIC_ASSETS.items():
        target = H4W_PATH / relative
        if target.is_file():
            print(f"[h4wpp-setup] {label}: already present")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[h4wpp-setup] downloading {label} from {url}")
        urllib.request.urlretrieve(url, target)


def check_assets() -> int:
    missing = []
    for label, relative in REQUIRED_ASSETS:
        path = H4W_PATH / relative
        if path.is_file() and path.stat().st_size > 0:
            print(f"[h4wpp-setup] OK   {label}: {path}")
        else:
            missing.append((label, relative))
            print(f"[h4wpp-setup] MISS {label}: {path}")
    if missing:
        print("\nMissing assets are intentionally not redistributed by this repository.")
        print("Provide them under --weights-root using the relative paths above, or")
        print("set up the corresponding official/licensed downloads and rerun setup.")
        return 2
    return 0


def check_runtime(python: str) -> int:
    modules = ("torch", "torchvision", "smplx", "pytorch3d", "ultralytics", "mmpose")
    code = (
        "import importlib.util, sys; "
        "mods=sys.argv[1:]; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print('MISSING:' + ','.join(missing) if missing else 'OK'); "
        "raise SystemExit(2 if missing else 0)"
    )
    result = subprocess.run([python, "-c", code, *modules], text=True, capture_output=True)
    print(f"[h4wpp-setup] runtime {python}: {result.stdout.strip() or result.stderr.strip()}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Initialize pinned third-party code and compatibility patch.")
    parser.add_argument("--check", action="store_true", help="Check pinned code and all required model assets.")
    parser.add_argument("--download-public", action="store_true", help="Download the public WiLoR weights from Hugging Face.")
    parser.add_argument("--weights-root", type=Path, help="Directory containing licensed assets in H4W++ relative layout.")
    parser.add_argument("--runtime-python", default=os.environ.get("REID_H4WPP_PYTHON", sys.executable))
    args = parser.parse_args()
    if not (args.install or args.check or args.download_public or args.weights_root):
        parser.error("choose --install, --check, --download-public, or --weights-root")

    if args.install:
        if not H4W_PATH.is_dir():
            run(["git", "submodule", "update", "--init", "--recursive", "third-party/Hand4Whole-plus-plus_RELEASE"], cwd=REPO_ROOT)
        wilor, _ = wire_dependencies()
        apply_wilor_patch(wilor)
    elif not H4W_PATH.is_dir():
        raise FileNotFoundError(f"H4W++ source is missing: {H4W_PATH}; run --install")

    if args.weights_root:
        copy_assets(args.weights_root.expanduser().resolve())
    if args.download_public:
        download_public_assets()
    if args.check:
        if git_head(H4W_PATH):
            print(f"[h4wpp-setup] H4W++ source: {git_head(H4W_PATH)}")
        if (DEPS_ROOT / "WiLoR").is_dir():
            print(f"[h4wpp-setup] WiLoR source: {git_head(DEPS_ROOT / 'WiLoR')}")
        if (DEPS_ROOT / "mmpose").is_dir():
            print(f"[h4wpp-setup] MMPose source: {git_head(DEPS_ROOT / 'mmpose')}")
        asset_status = check_assets()
        runtime_status = check_runtime(args.runtime_python)
        return max(asset_status, runtime_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
