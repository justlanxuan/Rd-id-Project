"""Official adapters for the three supported raw dataset layouts."""

from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter, PreprocessArtifact
from .prepared import validate_prepared_dataset
from .validation import validate_preprocess_output


class TotalCaptureAdapter(DatasetAdapter):
    dataset_name = "totalcapture"

    def preprocess(self, *, output_dir=None, manifest_csv=None) -> PreprocessArtifact:
        reused = _reuse_prepared(
            self,
            split_identity="subject",
            expected_test_key="test_subjects",
        )
        if reused is not None:
            return reused

        from preprocess.datasets.totalcapture import run_preprocess

        output = validate_preprocess_output(
            self.dataset_name,
            run_preprocess(self.config_path, output_dir=output_dir, manifest_csv=manifest_csv),
        )
        return PreprocessArtifact(self.dataset_name, output, _optional_path(manifest_csv))


class EgoHumansAdapter(DatasetAdapter):
    dataset_name = "egohumans"

    def preprocess(self, *, output_dir=None, manifest_csv=None) -> PreprocessArtifact:
        reused = _reuse_prepared(
            self,
            split_identity="session",
            expected_test_key="test_sessions",
        )
        if reused is not None:
            return reused

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
                split_identity="session",
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


def _reuse_prepared(
    adapter: DatasetAdapter,
    *,
    split_identity: str,
    expected_test_key: str,
) -> PreprocessArtifact | None:
    from preprocess.common.config import resolve_config

    config = resolve_config(adapter.config_path)
    preprocess_config = config.get("preprocess", {})
    if not bool(preprocess_config.get("reuse_prepared", False)):
        return None
    prepared_root = str(preprocess_config.get("prepared_root", "") or "").strip()
    if not prepared_root:
        raise ValueError("preprocess.reuse_prepared=true requires preprocess.prepared_root")
    slice_config = config.get("slice", {})
    expected_test_values = _as_set(slice_config.get(expected_test_key, []))
    frame_acc_config = config.get("test", {}).get("metrics", {}).get("frame_acc", {})
    allow_singletons = str(frame_acc_config.get("singleton_policy", "error")) == "exclude"
    prepared = validate_prepared_dataset(
        prepared_root,
        split_identity=split_identity,
        expected_test_values=expected_test_values or None,
        allow_singleton_test_groups=allow_singletons,
    )
    return PreprocessArtifact(adapter.dataset_name, prepared, None, prepared=True)


def _as_set(value) -> set[str]:
    if value is None or value == "":
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(item).strip() for item in value if str(item).strip()}
