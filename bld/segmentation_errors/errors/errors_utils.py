import numpy as np


class ErrorsUtils:
    @staticmethod
    def validate_magnitude(magnitude_mm: float) -> None:
        if not np.isfinite(magnitude_mm):
            raise ValueError("magnitude_mm must be finite.")
        if magnitude_mm < 0:
            raise ValueError("magnitude_mm must be non-negative.")

    @staticmethod
    def validate_direction(axis: int, direction: int) -> None:
        if not isinstance(axis, (int, np.integer)):
            raise TypeError("axis must be an integer.")
        if direction not in (-1, 1):
            raise ValueError("direction must be either +1 or -1.")
