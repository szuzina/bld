from __future__ import annotations
from typing import Union, Sequence

import numpy as np


class SegmentationError:
    """
    Store 2D result data.
    """
    def __init__(self, result_mask: np.ndarray, error_type: str, magnitude_mm: Union[float, Sequence[float]]):
        self.result_mask = result_mask
        self.error_type = error_type
        self.magnitude = magnitude_mm
