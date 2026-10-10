from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from bld.segmentation_errors import SegmentationError
from bld.segmentation_errors.errors.error_generator import ErrorGenerator


class CreateSegmentationError(ErrorGenerator):
    """Generate controlled errors in a binary segmentation mask (one image slice).

    Parameters
    ----------
    mask:
        NumPy array containing the binary segmentation mask.
    spacing:
        Physical spacing of the mask axes in mm, as
        (axis0_spacing, axis1_spacing).
    seed:
        Seed for reproducible random-boundary errors.

    Notes
    -----
    The original mask is never modified in place. All operations return a new NumPy array.
    """

    def __init__(self, mask: np.ndarray, spacing: Tuple[float, float],
                 error_type: str, magnitude_mm: Union[float, Sequence[float]], seed: Optional[int] = None):
        """
                error_type:
            One of ``expansion``, ``erosion``, ``translation``,
            ``directional_expansion``, ``directional_erosion`` or
            ``random_boundary``.
        magnitude_mm:
            Error magnitude in mm. For translation this may be a scalar or a vector.
            For directional errors it is a scalar.
        """
        super().__init__(mask=mask, spacing=spacing, error_type=error_type,
                         magnitude_mm=magnitude_mm, seed=seed)

        self.result = self.create_result()

    def create(self, **kwargs: Any) -> np.ndarray:

        if self.error_type not in self._ERROR_TYPES:
            raise ValueError(
                f"Unknown error_type '{self.error_type}'. Choose from "
                f"{sorted(self._ERROR_TYPES)}."
            )

        if self.error_type == "expansion":
            return self.expansion(float(self.magnitude_mm))
        elif self.error_type == "erosion":
            return self.erosion(float(self.magnitude_mm))
        elif self.error_type == "translation":
            return self.translation(self.magnitude_mm)
        elif self.error_type == "directional_expansion":
            return self.directional_expansion(
                magnitude_mm=float(self.magnitude_mm),
                axis=kwargs.get("axis", 0),
                direction=kwargs.get("direction", 1),
            )
        elif self.error_type == "directional_erosion":
            return self.directional_erosion(
                magnitude_mm=float(self.magnitude_mm),
                axis=kwargs.get("axis", 0),
                direction=kwargs.get("direction", 1),
            )
        else:
            return self.random_boundary(
                magnitude_mm=float(self.magnitude_mm),
                probability=kwargs.get("probability", 0.5),
            )

    def create_result(self, **kwargs: Any) -> SegmentationError:
        """Create an error and return both the mask and quantitative metadata as a SegmentationError class object."""
        result_contour = self.create(**kwargs)
        return SegmentationError(result_mask=result_contour,
                                 error_type=self.error_type,
                                 magnitude_mm=self.magnitude_mm)
