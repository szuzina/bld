from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage

from bld.segmentation_errors.errors.errors_utils import ErrorsUtils


class ErrorGenerator:
    """Generate controlled errors in a binary segmentation mask (one image slice).

    Parameters
    ----------
    mask:
        NumPy array containing the binary segmentation mask.
    spacing:
        Physical spacing of the mask axes in mm, as (spacing_y, spacing_x).
    seed:
        Seed for reproducible random-boundary errors.

    error_type:
        One of ``expansion``, ``erosion``, ``translation``,
        ``directional_expansion``, ``directional_erosion`` or
        ``random_boundary``.
    magnitude_mm:
        Error magnitude in mm. For translation this may be a scalar or a vector.

    Notes
    -----
    The original mask is never modified in place. All operations return a new NumPy array.
    """

    _ERROR_TYPES = {
        "expansion",
        "erosion",
        "translation",
        "directional_expansion",
        "directional_erosion",
        "random_boundary",
    }

    def __init__(self, mask: np.ndarray, spacing: Tuple[float, float],
                 error_type: str, magnitude_mm: Union[float, Sequence[float]], seed: Optional[int] = None):

        self.mask = mask
        self.spacing = spacing

        self.error_type = error_type.lower().strip()
        self.magnitude_mm = magnitude_mm

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Mask modification functions
    # ------------------------------------------------------------------

    def expansion(self, magnitude_mm: float) -> np.ndarray:
        """Expand the mask isotropically by ``magnitude_mm``.

        ndimage.binary_dilatation:
        Multidimensional binary dilation with the given structuring element.
        input:
            Binary array_like to be dilated. Non-zero (True) elements form the subset to be dilated.
        structure:
            Structuring element used for the dilation. Non-zero elements are considered True. If no structuring element
            is provided an element is generated with a square connectivity equal to one.
        """
        ErrorsUtils.validate_magnitude(magnitude_mm)
        if magnitude_mm == 0:
            return self.mask.copy()
        structure = ErrorGenerator.ellipse_structure(magnitude_mm, spacing=self.spacing)
        return ndimage.binary_dilation(input=self.mask, structure=structure).astype(self.mask.dtype)

    def erosion(self, magnitude_mm: float) -> np.ndarray:
        """Erode the mask isotropically by ``magnitude_mm``."""
        ErrorsUtils.validate_magnitude(magnitude_mm)
        if magnitude_mm == 0:
            return self.mask.copy()
        structure = ErrorGenerator.ellipse_structure(magnitude_mm, spacing=self.spacing)
        return ndimage.binary_erosion(input=self.mask, structure=structure)

    def translation(self, shift_mm: Union[float, Sequence[float]]) -> np.ndarray:
        """Translate a 2D binary segmentation mask by physical distance (in mm).

        A scalar applies the identical shift to both (y, x) axes.
        A 2-element sequence specifies (shift_y_mm, shift_x_mm).
        """

        if self.mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, but got {self.mask.ndim}D.")

        # Convert scalar to (shift, shift) or validate 2-element sequence
        if np.isscalar(shift_mm):
            shifts_mm = np.array([float(shift_mm), float(shift_mm)])
        else:
            shifts_mm = np.asarray(shift_mm, dtype=float)
            if shifts_mm.shape != (2,):
                raise ValueError(
                    f"shift_mm must contain exactly 2 values (y, x) for a 2D mask, "
                    f"got {shifts_mm.shape[0]}."
                )

        spacing = np.asarray(self.spacing, dtype=float)
        if spacing.shape != (2,):
            raise ValueError(
                f"self.spacing must contain exactly 2 values (spacing_y, spacing_x), "
                f"got {spacing.shape[0]}."
            )

        # Convert mm to integer pixel shifts
        shifts_vox = np.rint(shifts_mm / spacing).astype(int)

        # Apply 2D shift (shift=(shift_row, shift_col))
        translated = ndimage.shift(
            self.mask.astype(np.uint8),
            shift=(shifts_vox[0], shifts_vox[1]),
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )

        return translated.astype(bool)

    def directional_expansion(self, magnitude_mm: float, axis: int = 0, direction: int = 1) -> np.ndarray:
        """Expand the 2D binary mask only towards one side of an axis.

        ``axis`` must be 0 (vertical / rows) or 1 (horizontal / columns).
        ``direction=+1`` expands toward increasing index values (down or right).
        ``direction=-1`` expands toward decreasing index values (up or left).
        """

        if self.mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, but got {self.mask.ndim}D.")
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1 for a 2D mask, got {axis}.")

        ErrorsUtils.validate_direction(axis, direction)
        ErrorsUtils.validate_magnitude(magnitude_mm)
        if magnitude_mm == 0:
            return self.mask.copy()

        # Generate the isotropic candidate expansion
        expanded = self.expansion(magnitude_mm)
        new_pixels = expanded & ~self.mask

        coords = np.where(self.mask)
        if len(coords[0]) == 0:
            return self.mask.copy()

        # Find the leading edge coordinate along the target axis
        boundary = coords[axis].max() if direction > 0 else coords[axis].min()
        distance_px = int(np.ceil(magnitude_mm / self.spacing[axis]))

        # Memory-efficient coordinate grid for 2D using np.ogrid
        h, w = self.mask.shape
        grid_axis = np.ogrid[0:h, 0:w][axis]

        if direction > 0:
            side = (grid_axis > boundary) & (grid_axis <= boundary + distance_px)
        else:
            side = (grid_axis < boundary) & (grid_axis >= boundary - distance_px)

        return self.mask | (new_pixels & side)

    def directional_erosion(self, magnitude_mm: float, axis: int = 0, direction: int = 1) -> np.ndarray:
        """Remove segmentation only from one side of an axis for a 2D mask.

        ``axis`` must be 0 (vertical / rows) or 1 (horizontal / columns).
        ``direction=+1`` removes pixels from the higher index side (bottom or right).
        ``direction=-1`` removes pixels from the lower index side (top or left).
        """

        if self.mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, but got {self.mask.ndim}D.")
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1 for a 2D mask, got {axis}.")

        ErrorsUtils.validate_direction(axis, direction)
        ErrorsUtils.validate_magnitude(magnitude_mm)

        if magnitude_mm == 0:
            return self.mask.copy()

        eroded = self.erosion(magnitude_mm)
        removed = self.mask & ~eroded

        coords = np.where(self.mask)
        if len(coords[0]) == 0:
            return self.mask.copy()

        boundary = coords[axis].max() if direction > 0 else coords[axis].min()
        distance_px = int(np.ceil(magnitude_mm / self.spacing[axis]))

        # Memory-efficient coordinate grid for 2D via np.ogrid
        h, w = self.mask.shape
        grid_axis = np.ogrid[0:h, 0:w][axis]

        if direction > 0:
            side = grid_axis >= boundary - distance_px + 1
        else:
            side = grid_axis <= boundary + distance_px - 1

        return self.mask & ~(removed & side)

    def random_boundary(self, magnitude_mm: float, probability: float = 0.5) -> np.ndarray:
        """Randomly add/remove boundary pixels within a physical distance band for a 2D mask.

        ``probability`` is the probability of accepting each candidate boundary change.
        Uses ``self.rng`` seeded during class initialization.
        """
        if self.mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, but got {self.mask.ndim}D.")

        ErrorsUtils.validate_magnitude(magnitude_mm)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be between 0 and 1, got {probability}.")
        if magnitude_mm == 0:
            return self.mask.copy()

        # 2D structuring element scaled by physical spacing (spacing_y, spacing_x)
        structure = ErrorGenerator.ellipse_structure(magnitude_mm, spacing=self.spacing)

        outer_band = ndimage.binary_dilation(self.mask, structure=structure) & ~self.mask
        inner_band = self.mask & ~ndimage.binary_erosion(self.mask, structure=structure)

        result = self.mask.copy()
        if probability > 0:
            # Generate random candidate maps matching the 2D mask shape (H, W)
            add = self.rng.random(result.shape) < probability
            remove = self.rng.random(result.shape) < probability
            result |= (outer_band & add)
            result &= ~(inner_band & remove)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def ellipse_structure(radius_mm: float, spacing: Tuple[float, float]) -> np.ndarray:
        """Generate a 2D binary ellipse structuring element based on physical spacing."""
        radii_px = [radius_mm / s for s in spacing]
        y_max = int(np.ceil(radii_px[0]))
        x_max = int(np.ceil(radii_px[1]))

        y, x = np.ogrid[-y_max:y_max + 1, -x_max:x_max + 1]
        dist_sq = (y * spacing[0]) ** 2 + (x * spacing[1]) ** 2
        return dist_sq <= radius_mm ** 2
