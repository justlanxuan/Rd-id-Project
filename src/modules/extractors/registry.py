"""Extractor-domain registry."""

from __future__ import annotations

from typing import Any

from src.core import Registry

from .base import VideoSkeletonExtractor

EXTRACTOR_REGISTRY: Registry[VideoSkeletonExtractor] = Registry("extractor")


@EXTRACTOR_REGISTRY.register("alphapose_full")
def _build_alphapose_full(cfg: dict[str, Any]) -> VideoSkeletonExtractor:
    from .alphapose_full import AlphaPoseFullExtractor

    return AlphaPoseFullExtractor(cfg)


@EXTRACTOR_REGISTRY.register("bytetrack_alphapose")
def _build_bytetrack_alphapose(cfg: dict[str, Any]) -> VideoSkeletonExtractor:
    from .bytetrack_alphapose import ByteTrackAlphaPoseExtractor

    return ByteTrackAlphaPoseExtractor(cfg)


@EXTRACTOR_REGISTRY.register("wham")
def _build_wham(cfg: dict[str, Any]) -> VideoSkeletonExtractor:
    from .wham import WHAMExtractor

    return WHAMExtractor(cfg)


@EXTRACTOR_REGISTRY.register("hand4whole_pp")
def _build_hand4whole_pp(cfg: dict[str, Any]) -> VideoSkeletonExtractor:
    from .hand4whole_pp import Hand4WholePPExtractor

    return Hand4WholePPExtractor(cfg)


def build_extractor(
    name: str,
    cfg: dict[str, Any],
    *,
    allow_experimental: bool = False,
) -> VideoSkeletonExtractor:
    canonical_name = EXTRACTOR_REGISTRY.resolve_name(name)
    if canonical_name == "wham" and not allow_experimental:
        raise RuntimeError(
            "Extractor 'wham' is experimental and does not yet emit the canonical skeleton schema. "
            "Set allow_experimental=True only for isolated development runs."
        )
    return EXTRACTOR_REGISTRY.build(canonical_name, cfg)
