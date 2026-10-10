from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Sequence

import numpy as np


@dataclass
class SegmentationError:
    """
    Store 2D result data.
    """
    result_mask: np.ndarray
    error_type: str
    magnitude_mm: Union[float, Sequence[float]]
