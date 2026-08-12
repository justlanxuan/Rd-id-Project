"""Repository-owned preprocessing package."""

from .adapters import DATASET_ADAPTERS, DatasetAdapter, build_dataset_adapter

_DATASET_MODULES = ("custom", "custom_plus", "egohumans", "totalcapture")
__all__ = ["DATASET_ADAPTERS", "DatasetAdapter", "build_dataset_adapter", *_DATASET_MODULES]


def __getattr__(name: str):
    if name in _DATASET_MODULES:
        import importlib

        module = importlib.import_module(f"preprocess.datasets.{name}")
        return module
    raise AttributeError(name)
