from typing import Optional, List

import numpy as np

from bld.data import DataLoader
from bld.data import DataDownloader
from bld.evaluation.traditional_metrics import TraditionalMetricsCalculator
from bld.metrics import MSICalculator


class MetricsEvaluator:
    """
    Calculates the different metrics for all the image slices of one patient.

    Args:
        patient: patient number
        data_downloader: DataDownloader
        il: inside penalty level value
        ol: outside penalty level value

    Returns:
        dl: the DataLoader class for the selected patient which contains the patient data
        num_slices: the number of slices
        msindex: MSI values
        idx: the slice indices for which MSI was calculated
        dice: Dice index values
        jacc: Jaccard index values
        haus: Hausdorff distance values
    """

    def __init__(self, patient: int,
                 data_downloader: DataDownloader,
                 il: Optional[float] = 1, ol: Optional[float] = 1):
        self.patient = patient

        self.il = il
        self.ol = ol

        self.dl = DataLoader(patient=patient, data_downloader=data_downloader)
        self.folder = self.dl.folder

        # Get number of slices available
        num_slices_test = len([key for key in self.dl.mask_test if key.startswith('slice')])
        num_slices_ref = len([key for key in self.dl.c_ref if key.startswith('slice')])
        self.num_slices = min(num_slices_test,
                              num_slices_ref)  # Use minimum to avoid exceeding available slices

        self.msindex: list = []
        self.idx: list = []
        self.dice: list = []
        self.jacc: list = []
        self.haus: list = []

        self.msi_with_zeros: list = []
        self.dice_all_slices: list = []
        self.jaccard_all_slices: list = []
        self.hausdorff_all_slices: list = []
        self.idx_all_slices: list = []

    @staticmethod
    def check_contours_on_slice(test_points: np.ndarray, ref_points: np.ndarray) -> bool:
        """
        Check if the reference and test contours are compatible and have at least one element.
        """
        if len(test_points) != len(ref_points) or len(test_points) == 0 or len(ref_points) == 0:
            error = True
        else:
            # Check if each array within test_points and ref_points is 2D
            for test_contour, ref_contour in zip(test_points, ref_points):
                if test_contour.ndim != 2 or ref_contour.ndim != 2:
                    error = True
                    return error  # Return immediately if an error is found
            error = False

        return error

    def find_msi_for_one_slice(self, slice_index: int) -> List:
        """
        Calculate MSI and traditional metrics for one image slice.
        """
        slice_name = 'slice' + str(slice_index)
        # Finding the MSI
        points_ref = self.dl.c_ref[slice_name]
        points_test = self.dl.c_test[slice_name]
        msi_calc = MSICalculator(
            il=self.il, ol=self.ol,
            ref_points=points_ref,
            test_points=points_test)
        msi_calc.run()

        return msi_calc.msi

    def find_traditional_metrics_for_one_slice(self, slice_index: int) -> TraditionalMetricsCalculator:
        """
        Calculate traditional metrics for one image slice (MSI is zero).
        """
        slice_name = 'slice' + str(slice_index)

        points_ref = self.dl.c_ref[slice_name]
        points_test = self.dl.c_test[slice_name]

        trad_metrics_calc = TraditionalMetricsCalculator(
                                points_ref=points_ref,
                                points_test=points_test,
                                slice_mask_ref=self.dl.mask_ref[slice_name],
                                slice_mask_test=self.dl.mask_test[slice_name])

        return trad_metrics_calc

    def evaluate(self):
        """
        Calculate the metrics for all image slices.
        """
        for i in range(self.num_slices):
            points_ref = self.dl.c_ref['slice' + str(i)]
            points_test = self.dl.c_test['slice' + str(i)]

            is_run_correctly = self.check_contours_on_slice(
                test_points=points_test,
                ref_points=points_ref)

            if not is_run_correctly:  # there is no error while checking the contours
                m = self.find_msi_for_one_slice(slice_index=i)
                t = self.find_traditional_metrics_for_one_slice(slice_index=i)
                self.msindex.append(m)
                self.idx.append(i)
                self.msi_with_zeros.append(m)
                self.dice_all_slices.append(t.dice)
                self.jaccard_all_slices.append(t.jaccard)
                self.hausdorff_all_slices.append(t.hausdorff)
                self.idx_all_slices.append(i)

            else:  # there was some kind of error while checking the contours (empty slice or incorrect pairing)
                # we still want to have the slice with traditional metrics and MSI=0
                t_2 = self.find_traditional_metrics_for_one_slice(slice_index=i)
                if len(points_ref) != 0 and len(points_test) != 0:  # there is at least one ref and one test point
                    # if there is only ref or only test contour, then all metrics will equal to zero/inf
                    # --> not interesting
                    self.msi_with_zeros.append(0)
                    self.dice_all_slices.append(t_2.dice)
                    self.jaccard_all_slices.append(t_2.jaccard)
                    self.hausdorff_all_slices.append(t_2.hausdorff)
                    self.idx_all_slices.append(i)
