"""WHAM 3D pose estimator adapter."""

import os
import sys
from typing import Dict, Optional


class WHAM3DConfig:
    """WHAM 3D estimator configuration"""
    def __init__(self, config_dict):
        self.repo_root = str(config_dict.get('repo_root', '') or '')
        if not self.repo_root:
            raise ValueError("WHAM3DConfig.repo_root is required")
        self.checkpoint_file = config_dict.get('checkpoint_file', None)
        self.device = config_dict.get('device', 'cuda:0')
        self.run_global = config_dict.get('run_global', True)
        self.output_dir = config_dict.get('output_dir', './wham_outputs')


class WHAM3DEstimator:
    """WHAM 3D human estimator. Receives video path, outputs 3D human mesh and parameters."""

    def __init__(self, config_dict: Dict):
        self.config = WHAM3DConfig(config_dict)
        self._wham_api = None
        self._initialized = False

    def _load_wham(self):
        """Lazy load WHAM model."""
        if self._initialized:
            return


        wham_path = self.config.repo_root
        if wham_path not in sys.path:
            sys.path.insert(0, wham_path)

        try:
            from wham_api import WHAM_API
            self._wham_api = WHAM_API()
            self._initialized = True
        except Exception as exc:
            raise RuntimeError(f"Failed to load WHAM from {wham_path}: {exc}") from exc

    def reset(self):
        """Reset estimator state."""
        pass

    def process_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """Process video file and obtain 3D human results."""
        self._load_wham()

        if output_dir is None:
            output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"[WHAM3D] Processing video: {video_path}")
        print(f"[WHAM3D] Output directory: {output_dir}")
        print(f"[WHAM3D] Device: {self.config.device}")
        print(f"[WHAM3D] Run global: {self.config.run_global}")

        try:
            results, tracking_results, slam_results = self._wham_api(
                video_path,
                output_dir=output_dir,
                run_global=self.config.run_global,
                visualize=False
            )

            print("[WHAM3D] Processing completed!")
            print(f"[WHAM3D] Detected {len(results)} persons")

            return {
                'results_3d': results,
                'tracking_results': tracking_results,
                'slam_results': slam_results
            }

        except Exception as exc:
            raise RuntimeError(f"WHAM processing failed: {exc}") from exc

    def __call__(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """Callable interface."""
        return self.process_video(video_path, output_dir)


def build_wham_3d_estimator(config_dict: Dict) -> WHAM3DEstimator:
    """Build WHAM 3D estimator."""
    return WHAM3DEstimator(config_dict)
