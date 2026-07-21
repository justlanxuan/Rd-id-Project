"""Backward-compatible imports for the alignment dataset.

New code should import from `src.datasets.alignment` and
`src.datasets.transforms`.
"""

from src.datasets.alignment import WindowAlignmentDataset
from src.datasets.transforms import lowpass_filter_fft, single_sensor_to_48d

__all__ = [
    "WindowAlignmentDataset",
    "lowpass_filter_fft",
    "single_sensor_to_48d",
]
