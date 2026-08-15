"""Dataset-specific preprocess entrypoints exposed from the repository root."""

__all__ = ["custom", "egohumans", "totalcapture"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        module = importlib.import_module(f"preprocess.datasets.{name}")
        return module
    raise AttributeError(name)
