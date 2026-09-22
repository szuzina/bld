"""Controlled generation of segmentation errors.

The `CreateSegmentationError` class creates reproducible, physically scaled perturbations of binary
2D segmentation masks.  Distances are specified in millimetres, so the generated error is independent of voxel size.

The class is intended for controlled segmentation-error experiments, e.g. relating a known contour perturbation to
segmentation metrics and downstream radiotherapy dose metrics.

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from bld.data.dataloader import DataLoader
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
from scipy import ndimage


class SegmentationError:
    """
    Store 2D result data.
    """
    def __init__(self, result_mask: np.ndarray, error_type: str, magnitude_mm: Union[float, Sequence[float]]):
        self.result_mask = result_mask
        self.error_type = error_type
        self.magnitude = magnitude_mm


class CreateSegmentationError:
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
        """
                error_type:
            One of ``expansion``, ``erosion``, ``translation``,
            ``directional_expansion``, ``directional_erosion`` or
            ``random_boundary``.
        magnitude_mm:
            Error magnitude in mm. For translation this may be a scalar or a vector.
            For directional errors it is a scalar.
        """
        self.mask = mask
        self.spacing = spacing

        self.error_type = error_type.lower().strip()
        self.magnitude_mm = magnitude_mm

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.result = self.create_result()

    def create(self, **kwargs: Any) -> np.ndarray:

        if self.error_type not in self._ERROR_TYPES:
            raise ValueError(
                f"Unknown error_type '{self.error_type}'. Choose from "
                f"{sorted(self._ERROR_TYPES)}."
            )

        if self.error_type == "expansion":
            return self.expansion(float(self.magnitude_mm))
        if self.error_type == "erosion":
            return self.erosion(float(self.magnitude_mm))
        if self.error_type == "translation":
            return self.translation(self.magnitude_mm)
        if self.error_type == "directional_expansion":
            return self.directional_expansion(
                magnitude_mm=float(self.magnitude_mm),
                axis=kwargs.get("axis", 0),
                direction=kwargs.get("direction", 1),
            )
        if self.error_type == "directional_erosion":
            return self.directional_erosion(
                magnitude_mm=float(self.magnitude_mm),
                axis=kwargs.get("axis", 0),
                direction=kwargs.get("direction", 1),
            )
        return self.random_boundary(
            magnitude_mm=float(self.magnitude_mm),
            probability=kwargs.get("probability", 0.5),
        )

    def create_result(self, **kwargs: Any) -> SegmentationError:
        """Create an error and return both the mask and quantitative metadata as a SegmentationError class object."""
        result_contour = self.create(**kwargs)
        return SegmentationError(result_mask=result_contour, error_type=self.error_type, magnitude_mm=self.magnitude_mm)

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
        self._validate_magnitude(magnitude_mm)
        if magnitude_mm == 0:
            return self.mask.copy()
        structure = self._ellipse_structure(magnitude_mm)
        return ndimage.binary_dilation(input=self.mask, structure=structure).astype(self.mask.dtype)

    def erosion(self, magnitude_mm: float) -> np.ndarray:
        """Erode the mask isotropically by ``magnitude_mm``."""
        self._validate_magnitude(magnitude_mm)
        if magnitude_mm == 0:
            return self.mask.copy()
        structure = self._ellipse_structure(magnitude_mm)
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

        self._validate_direction(axis, direction)
        self._validate_magnitude(magnitude_mm)
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

        self._validate_direction(axis, direction)
        self._validate_magnitude(magnitude_mm)

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

        self._validate_magnitude(magnitude_mm)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be between 0 and 1, got {probability}.")
        if magnitude_mm == 0:
            return self.mask.copy()

        # 2D structuring element scaled by physical spacing (spacing_y, spacing_x)
        structure = self._ellipse_structure(magnitude_mm)

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
    def _ellipse_structure(self, radius_mm: float) -> np.ndarray:
        """Generate a 2D binary ellipse structuring element based on physical spacing."""
        radii_px = [radius_mm / s for s in self.spacing]
        y_max = int(np.ceil(radii_px[0]))
        x_max = int(np.ceil(radii_px[1]))

        y, x = np.ogrid[-y_max:y_max + 1, -x_max:x_max + 1]
        dist_sq = (y * self.spacing[0]) ** 2 + (x * self.spacing[1]) ** 2
        return dist_sq <= radius_mm ** 2

    @staticmethod
    def _validate_magnitude(magnitude_mm: float) -> None:
        if not np.isfinite(magnitude_mm):
            raise ValueError("magnitude_mm must be finite.")
        if magnitude_mm < 0:
            raise ValueError("magnitude_mm must be non-negative.")

    @staticmethod
    def _validate_direction(axis: int, direction: int) -> None:
        if not isinstance(axis, (int, np.integer)):
            raise TypeError("axis must be an integer.")
        if direction not in (-1, 1):
            raise ValueError("direction must be either +1 or -1.")


class SegmentationErrorPatient:
    """Result of a segmentation-error operation on patient-level.

    Attributes
    ----------
    dl:
        DataLoader object of the selected patient.

    Notes
    ---------------
     mod_contours:
        Dictionary with the slice number and the corresponding modified binary contour (as a NumPy array).
    """

    def __init__(self, dl: DataLoader, error_type: str, magnitude_mm: Union[float, Sequence[float]],
                 seed: Optional[int] = None):
        self.dl = dl

        self.error_type = error_type
        self.magnitude_mm = magnitude_mm
        self.seed = seed

        self.results, self.metadata = self.generate_all_slices_for_one_patient()

    def generate_all_slices_for_one_patient(self, **kwargs: Any):
        mod_contours = {}
        for i in range(len(self.dl.c_ref)):
            mod_contours['slice' + str(i)] = CreateSegmentationError(mask=self.dl.mask_ref['slice' + str(i)],
                                                                     spacing=self.dl.spacing[:2],  # dl.spacing: (x,y,z)
                                                                     seed=0,
                                                                     error_type=self.error_type,
                                                                     magnitude_mm=self.magnitude_mm).result

        # generate metadata
        voxel_volume = float(np.prod(self.dl.spacing))

        original_volume = 0
        for i in range(len(self.dl.mask_ref)):
            original_volume += float(self.dl.mask_ref['slice' + str(i)].sum() * voxel_volume)
        modified_volume = 0
        for i in range(len(mod_contours)):
            modified_volume += float(mod_contours['slice' + str(i)].result_mask.sum() * voxel_volume)

        volume_change = modified_volume - original_volume
        percent = (
            100.0 * volume_change / original_volume
            if original_volume > 0
            else np.nan
        )

        metadata = {
            "error_type": self.error_type,
            "magnitude_mm": self._json_value(self.magnitude_mm),
            "spacing_mm": list(self.dl.spacing),
            "seed": self.seed,

            "original_volume_mm3": original_volume,
            "modified_volume_mm3": modified_volume,
            "volume_change_mm3": volume_change,
            "volume_change_percent": percent,
            **{k: self._json_value(v) for k, v in kwargs.items()},
        }

        return mod_contours, metadata

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value


class GenerateSeries:
    """Generate a reproducible series of errors.
    Results: a dictionary keyed as "<error_type>_<magnitude>mm", values: SegmentationError class objects."""

    def __init__(self, error_types: Iterable[str], magnitudes_mm: Iterable[float]):
        self.error_types = error_types
        self.magnitudes_mm = magnitudes_mm

        self.results = self.generate_series()

    def generate_series(self, **kwargs: Any) -> Dict[str, SegmentationErrorPatient]:
        results = {}
        for error in self.error_types:
            for magnitude in self.magnitudes_mm:
                result = SegmentationErrorPatient(
                    error_type=error,
                    magnitude_mm=magnitude,
                    **kwargs
                )
                key = f"{error}_{self._format_magnitude(magnitude)}mm"
                results[key] = result
        return results

    @staticmethod
    def _format_magnitude(value: float) -> str:
        return f"{float(value):g}"


class CreateVisualization:
    """Provides 2D visualization utilities for ground truth and modified segmentation masks."""

    def __init__(self, segmentations: Union[SegmentationError, Dict[str, SegmentationError]],
                 original_mask: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        segmentations : SegmentationError or Dict[str, SegmentationError]
            A single error result instance or a dictionary of series results.
        original_mask : np.ndarray, optional
            The reference ground truth binary mask (2D). If omitted, can be passed
            directly to visualization methods.
        """
        self.segmentations = segmentations
        self.original_mask = original_mask

    def show_comparison(self, result: Optional[SegmentationError] = None,
                        reference_mask: Optional[np.ndarray] = None,
                        figsize: Tuple[int, int] = (14, 5)) -> None:
        """Plot a 3-panel comparison: Reference, Perturbed mask, and Difference Map.

        Difference map color code:
        - Green: True Positive (Overlap)
        - Red: False Positive (Spurious expansion/addition)
        - Blue: False Negative (Missing/eroded region)
        """
        res = result or (
            self.segmentations
            if isinstance(self.segmentations, SegmentationError)
            else next(iter(self.segmentations.values()))
        )
        ref = reference_mask if reference_mask is not None else self.original_mask

        if ref is None:
            raise ValueError(
                "A reference (original) mask must be provided to show comparisons."
            )

        perturbed = res.result_mask

        # Build RGB difference overlay:
        # Green = TP, Red = FP, Blue = FN
        tp = ref & perturbed
        fp = perturbed & ~ref
        fn = ref & ~perturbed

        diff_map = np.zeros((*ref.shape, 3), dtype=np.uint8)
        diff_map[tp] = [46, 204, 113]  # Green
        diff_map[fp] = [231, 76, 60]   # Red
        diff_map[fn] = [52, 152, 219]  # Blue

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(ref, cmap="gray")
        axes[0].set_title("Ground Truth Mask")
        axes[0].axis("off")

        axes[1].imshow(perturbed, cmap="gray")
        axes[1].set_title("Modified Mask")
        axes[1].axis("off")

        axes[2].imshow(diff_map)
        axes[2].set_title("Diff: Green=TP, Red=FP, Blue=FN")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    def show_contour_overlay(self, result: Optional[SegmentationError] = None,
                             reference_mask: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (7, 7)) -> None:
        """Overlay boundaries of both masks on a single axes."""
        res = result or (
            self.segmentations
            if isinstance(self.segmentations, SegmentationError)
            else next(iter(self.segmentations.values()))
        )
        ref = reference_mask if reference_mask is not None else self.original_mask

        fig, ax = plt.subplots(figsize=figsize)
        if ref is not None:
            ax.result_mask(ref, levels=[0.5], colors=["lime"], linewidths=2)

        ax.result_mask(res.result_mask, levels=[0.5], colors=["red"], linestyles="dashed", linewidths=2)

        # Proxy lines for legend
        ax.plot([], [], color="lime", label="Reference")
        ax.plot([], [], color="red", linestyle="--", label="Perturbed")
        ax.legend(loc="upper right")
        ax.axis("off")
        ax.set_title("Contour Boundary Comparison")

        plt.tight_layout()
        plt.show()
