from __future__ import annotations
from typing import Union, Sequence, Optional, Any

import nibabel as nib
import numpy as np
import os

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
        for i in range(len(self.dl.mask_ref)):
            mod_contours['slice' + str(i)] = CreateSegmentationError(
                mask=self.dl.mask_ref['slice' + str(i)],
                spacing=self.dl.spacing[:2][::-1],  # dl.spacing: (x,y,z), we need (y,x)
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

    def save_masks_as_nifti(self):
        len_x = self.dl.mask_ref['slice0'].shape[0]
        len_y = self.dl.mask_ref['slice0'].shape[1]
        len_z = len(self.dl.mask_ref)
        modified_mask_patient = np.zeros((len_x, len_y, len_z))
        for i in range(len_z):
            modified_mask_patient[:, :, i] = self.results['slice' + str(i)].result_mask.T

        # Load original NIfTI
        original = nib.load(self.dl.labels_ref[self.dl.patient-1])
        # patient = 1 corresponds to the index 0 in the labels list

        # Create new NIfTI using the original spatial information
        new_img = nib.Nifti1Image(
            modified_mask_patient,
            affine=original.affine,
            header=original.header.copy(),
        )

        # Save
        if not os.path.isdir(os.path.join(self.dl.folder, 'segmentation_error_masks')):
            os.makedirs(os.path.join(self.dl.folder, 'segmentation_error_masks'), exist_ok=True)
        nifti_path = os.path.join(self.dl.folder, 'segmentation_error_masks')
        nifti_name = (
            f"patient{self.dl.patient}_"
            f"segmentation_error_"
            f"{self.error_type}_"
            f"{self.magnitude_mm}mm.nii.gz"
        )
        nib.save(new_img, os.path.join(nifti_path, nifti_name))

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
