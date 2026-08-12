"""Video-skeleton extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractorCapabilities:
    output_format: str
    dimensions: int
    experimental: bool = False


class VideoSkeletonExtractor(ABC):
    capabilities: ExtractorCapabilities

    @abstractmethod
    def extract(self, video_path: str, output_dir: str) -> str:
        raise NotImplementedError
