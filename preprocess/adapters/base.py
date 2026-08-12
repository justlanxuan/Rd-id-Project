"""Dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessArtifact:
    dataset: str
    output_dir: Path
    manifest_csv: Path | None = None
    prepared: bool = False


class DatasetAdapter(ABC):
    """Convert one raw dataset layout into the canonical sequence format."""

    dataset_name: str

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Dataset adapter config not found: {self.config_path}")

    @abstractmethod
    def preprocess(
        self,
        *,
        output_dir: str | Path | None = None,
        manifest_csv: str | Path | None = None,
    ) -> PreprocessArtifact:
        raise NotImplementedError
