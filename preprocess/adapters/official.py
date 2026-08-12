"""Official adapters for the three supported raw dataset layouts."""

from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter, PreprocessArtifact
from .prepared import validate_prepared_dataset
from .validation import validate_preprocess_output


class TotalCaptureAdapter(DatasetAdapter):
    dataset_name = "totalcapture"

    def preprocess(self, *, output_dir=None, manifest_csv=None) -> PreprocessArtifact:
        from preprocess.datasets.totalcapture import run_preprocess

        output = validate_preprocess_output(
            self.dataset_name,
            run_preprocess(self.config_path, output_dir=output_dir, manifest_csv=manifest_csv),
        )
        return PreprocessArtifact(self.dataset_name, output, _optional_path(manifest_csv))


class EgoHumansAdapter(DatasetAdapter):
    dataset_name = "egohumans"

    def preprocess(self, *, output_dir=None, manifest_csv=None) -> PreprocessArtifact:
        from preprocess.datasets.egohumans import run_preprocess

        output = validate_preprocess_output(
            self.dataset_name,
            run_preprocess(self.config_path, output_dir=output_dir, manifest_csv=manifest_csv),
        )
        return PreprocessArtifact(self.dataset_name, output, _optional_path(manifest_csv))


class CustomAdapter(DatasetAdapter):
    dataset_name = "custom"

    def preprocess(self, *, output_dir=None, manifest_csv=None) -> PreprocessArtifact:
        from preprocess.common.config import resolve_config

        config = resolve_config(self.config_path)
        preprocess_config = config.get("preprocess", {})
        prepared_root = str(preprocess_config.get("prepared_root", "") or "").strip()
        if bool(preprocess_config.get("reuse_prepared", False)):
            if not prepared_root:
                raise ValueError("preprocess.reuse_prepared=true requires preprocess.prepared_root")
            slice_config = config.get("slice", {})
            test_sessions = _as_set(slice_config.get("test_sessions", []))
            frame_acc_config = config.get("test", {}).get("metrics", {}).get("frame_acc", {})
            allow_singletons = str(frame_acc_config.get("singleton_policy", "error")) == "exclude"
            prepared = validate_prepared_dataset(
                prepared_root,
                expected_test_sessions=test_sessions or None,
                allow_singleton_test_groups=allow_singletons,
            )
            return PreprocessArtifact(self.dataset_name, prepared, None, prepared=True)

        from preprocess.datasets.custom import run_preprocess

        output = validate_preprocess_output(
            self.dataset_name,
            run_preprocess(self.config_path, output_dir=output_dir, manifest_csv=manifest_csv),
        )
        return PreprocessArtifact(self.dataset_name, output, _optional_path(manifest_csv))


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value).expanduser().resolve() if value is not None and str(value).strip() else None


def _as_set(value) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(item).strip() for item in value if str(item).strip()}
