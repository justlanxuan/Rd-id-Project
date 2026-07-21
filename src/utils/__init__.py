"""Utility helpers.

Keep this package import lightweight. Heavy helpers are imported lazily so
configuration loading does not require numpy/scipy.
"""


def __getattr__(name):
    if name == "run_chunk_trials":
        from .chunk_matcher import run_chunk_trials

        return run_chunk_trials
    raise AttributeError(name)


__all__ = ["run_chunk_trials"]
