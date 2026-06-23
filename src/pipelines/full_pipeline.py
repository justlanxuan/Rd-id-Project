"""Full pipeline driver."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.pipelines.stages import PreprocessStage, SliceStage, ExtractStage, TrainStage, TestStage
from src.utils.config import resolve_config


class FullPipeline:
    """Drive the full or partial workflow from a config file."""

    AVAILABLE_STAGES = {
        "preprocess": PreprocessStage,
        "slice": SliceStage,
        "extract": ExtractStage,
        "train": TrainStage,
        "test": TestStage,
    }

    def __init__(self, config_path: str, stages: List[str] | None = None):
        self.config_path = Path(config_path).expanduser().resolve()
        # Default full pipeline includes preprocess; extract must run before slice
        # for video-based workflows. ExtractStage simply skips when no extract section.
        self.stages = stages or ["preprocess", "extract", "slice", "train", "test"]
        for name in self.stages:
            if name not in self.AVAILABLE_STAGES:
                raise ValueError(f"Unknown stage: {name}. Available: {list(self.AVAILABLE_STAGES.keys())}")

    def run(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"config_path": self.config_path}
        cfg = resolve_config(self.config_path)
        test_cfg = cfg.get("test", {}) if isinstance(cfg, dict) else {}
        matcher_cfg = test_cfg.get("matcher", {}) if isinstance(test_cfg, dict) else {}
        physics_cfg = matcher_cfg.get("physics_based_matcher", {}) if isinstance(matcher_cfg, dict) else {}
        dl_cfg = matcher_cfg.get("dl_matcher", {}) if isinstance(matcher_cfg, dict) else {}
        skip_train = bool(physics_cfg.get("enabled", False)) and not bool(dl_cfg.get("enabled", True))

        print(f"[Pipeline] Config: {self.config_path}")
        print(f"[Pipeline] Stages : {self.stages}")
        for name in self.stages:
            if skip_train and name == "train":
                print("\n========== Stage: train ==========")
                print("[INFO] Physics-only matcher selected; skipping train stage.")
                continue
            print(f"\n========== Stage: {name} ==========")
            stage = self.AVAILABLE_STAGES[name]({})
            state = stage.run(state)
        print("\n========== Pipeline finished ==========")
        return state
