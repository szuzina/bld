from __future__ import annotations
from typing import Union, Sequence, Optional, Any

import numpy as np

from bld.data import DataLoader
from bld.segmentation_errors import CreateSegmentationError


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

    def __init__(self, dl: DataLoader,
                 error_type: str,
                 magnitude_mm: Union[float, Sequence[float]],
                 seed: Optional[int] = None):
        self.dl = dl

        self.error_type = error_type
        self.magnitude_mm = magnitude_mm
        self.seed = seed

        self.results, self.metadata = self.generate_all_slices_for_one_patient()

    def generate_all_slices_for_one_patient(self, **kwargs: Any):
        mod_contours = {}
        for i in range(len(self.dl.c_ref)):
            mod_contours['slice' + str(i)] = CreateSegmentationError(
                mask=self.dl.mask_ref['slice' + str(i)],
                spacing=self.dl.spacing[:2],  # dl.spacing: (x,y,z)
                seed=0,
                error_type=self.error_type,
                magnitude_mm=self.magnitude_mm
            ).result

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
