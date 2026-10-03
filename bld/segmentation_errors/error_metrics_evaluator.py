from typing import Optional, List

import cv2 as cv
import csv
import SimpleITK as SITK
from natsort import natsorted
import numpy as np
import os
from pathlib import Path

from bld.data import DataLoader
from bld.evaluation.traditional_metrics import TraditionalMetricsCalculator
from bld.metrics import MSICalculator, EvaluationMetrics


class ErrorMetricsEvaluator:
    """
    Calculates the different metrics for all the image slices of one patient.
    For one reference segmentation, metrics are calculated for many perturbed segmentations.

    Args:
        dl: DataLoader
        error_dir: the directory path containing the perturbed contours
        il: inside penalty level value for MSI
        ol: outside penalty level value for MSI

    Returns:
        num_slices: the number of slices
        msindex: MSI values
        idx: the slice indices for which MSI was calculated
        dice: Dice index values
        jacc: Jaccard index values
        haus: Hausdorff distance values
    """

    def __init__(self,
                 error_dir: str,
                 dl: DataLoader,
                 il: Optional[float] = 1, ol: Optional[float] = 1):

        self.il = il
        self.ol = ol

        self.dl = dl

        # Get number of slices available
        num_slices_test = len([key for key in self.dl.mask_test if key.startswith('slice')])
        num_slices_ref = len([key for key in self.dl.c_ref if key.startswith('slice')])
        self.num_slices = min(num_slices_test,
                              num_slices_ref)  # Use minimum to avoid exceeding available slices

        self.results = self.evaluate_all_tests(dir_path=error_dir)

    def find_msi_for_one_slice(self, slice_index: int, points_test_slice: np.ndarray[int]) -> List:
        """
        Calculate MSI and traditional metrics for one image slice.
        """
        slice_name = 'slice' + str(slice_index)
        # Finding the MSI
        points_ref = self.dl.c_ref[slice_name]
        points_test = points_test_slice
        msi_calc = MSICalculator(
            il=self.il, ol=self.ol,
            ref_points=points_ref,
            test_points=points_test)
        msi_calc.run()

        return msi_calc.msi

    def find_traditional_metrics_for_one_slice(self, slice_index: int,
                                               points_test_slice: np.ndarray[int],
                                               mask_test_dict: dict) \
            -> TraditionalMetricsCalculator:
        """
        Calculate traditional metrics for one image slice (MSI is zero).
        """
        slice_name = 'slice' + str(slice_index)

        ref_points = self.dl.c_ref[slice_name]
        test_points = points_test_slice

        trad_metrics_calc = TraditionalMetricsCalculator(
                                points_ref=ref_points,
                                points_test=test_points,
                                slice_mask_ref=self.dl.mask_ref[slice_name],
                                slice_mask_test=mask_test_dict[slice_name])

        return trad_metrics_calc

    def evaluate_one_test(self, test_path):
        """
        Calculate the metrics for all image slices of one test segmentation.
        """

        # find the test contour from the perturbed segmentation --> for MSI calculation
        test_contour_perturbed = self.get_contour_from_image(file_path=test_path)
        # find the test masks from the perturbed segmentation --> for traditional metrics calculation
        masks_test = self.get_mask(path=test_path)

        # initialize
        msindex: list = []
        idx: list = []
        dice: list = []
        jacc: list = []
        haus: list = []

        msi_with_zeros: list = []
        dice_all_slices: list = []
        jaccard_all_slices: list = []
        hausdorff_all_slices: list = []
        idx_all_slices: list = []

        for i in range(self.num_slices):
            points_ref = self.dl.c_ref['slice' + str(i)]
            points_test = test_contour_perturbed['slice' + str(i)]

            is_run_correctly = self.check_contours_on_slice(
                test_points=points_test,
                ref_points=points_ref)

            if not is_run_correctly:  # there is no error while checking the contours
                m = self.find_msi_for_one_slice(slice_index=i, points_test_slice=points_test)
                t = self.find_traditional_metrics_for_one_slice(slice_index=i,
                                                                points_test_slice=points_test,
                                                                mask_test_dict=masks_test)

                msindex.append(m)
                idx.append(i)
                dice.append(t.dice)
                jacc.append(t.jaccard)
                haus.append(t.hausdorff)

                msi_with_zeros.append(m)
                dice_all_slices.append(t.dice)
                jaccard_all_slices.append(t.jaccard)
                hausdorff_all_slices.append(t.hausdorff)
                idx_all_slices.append(i)

            else:  # there was some kind of error while checking the contours (empty slice or incorrect pairing)
                # we still want to have the slice with traditional metrics and MSI=0
                t_2 = self.find_traditional_metrics_for_one_slice(slice_index=i,
                                                                  points_test_slice=points_test,
                                                                  mask_test_dict=masks_test)
                if len(points_ref) != 0 and len(points_test) != 0:  # there is at least one ref and one test point
                    # if there is only ref or only test contour, then all metrics will equal to zero/inf
                    # --> not interesting
                    msi_with_zeros.append(0)
                    dice_all_slices.append(t_2.dice)
                    jaccard_all_slices.append(t_2.jaccard)
                    hausdorff_all_slices.append(t_2.hausdorff)
                    idx_all_slices.append(i)

        evaluation_metrics = EvaluationMetrics(msi=msindex, dice=dice, jaccard=jacc, hausdorff=haus, idx=idx)
        evaluation_metrics_with_zeros = EvaluationMetrics(msi=msi_with_zeros, dice=dice_all_slices,
                                                          jaccard=jaccard_all_slices, hausdorff=hausdorff_all_slices,
                                                          idx=idx_all_slices)

        return evaluation_metrics, evaluation_metrics_with_zeros

    def evaluate_all_tests(self, dir_path):
        error_results_metrics = {}
        path_to_dir = Path(dir_path)
        for item in natsorted(path_to_dir.glob("*.nii.gz")):
            print("start", item)
            error_results_metrics[item.name] = self.evaluate_one_test(item)

        return error_results_metrics

    @staticmethod
    def get_mask(path: str):
        """
        Creates a dictionary for a patient, contains the slice masks in np array.
        """

        mask_t = SITK.ReadImage(fileName=path)
        imgs = SITK.GetArrayFromImage(image=mask_t)
        number_of_slices = imgs.shape[0]

        masks = {}
        for i in range(number_of_slices):
            masks['slice' + str(i)] = imgs[i, :, :]

        return masks

    def get_contour_from_image(self, file_path: str) -> dict:
        """
        Converts a nii.gz image to a list of contours.

        Args:
            file_path: the path of the selected nii.gz image

        Returns:
          dictionary:
            keys - slice number (starting from 0)
            values - contours of the corresponding slice, each contour is one 2D numpy array
            with the coordinates of the contour points
        """

        im = SITK.ReadImage(fileName=file_path)
        img = SITK.GetArrayFromImage(image=im)

        # initialize dictionary
        dictionary_contours = dict()
        # get the contours
        for i in range(img.shape[0]):
            f_path = os.path.join(self.dl.folder, 'image.png')
            cv.imwrite(f_path, img[i] * 255)
            image = cv.imread(f_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            edged = cv.Canny(gray, 30, 200)
            contours, hierarchy = cv.findContours(
                edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            c = []
            for contour in contours:
                c.append(contour.T.squeeze())
            dictionary_contours['slice' + str(i)] = c

        return dictionary_contours

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

    def save_results_as_csv(self, output_path: str) -> None:
        """Save the evaluation results to a CSV file.

        One row is created for each evaluated slice of each perturbed
        segmentation. Both metric versions, with and without zero-filled
        slices, are included.

        Parameters
        ----------
        output_path:
            Path of the output CSV file.
        """

        fieldnames = [
            "test_segmentation",
            "slice_index",
            "include_zeros",
            "msi",
            "dice",
            "jaccard",
            "hausdorff",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for test_segmentation, results in self.results.items():
                evaluation_metrics, evaluation_metrics_with_zeros = results

                # Metrics without zeros
                for idx, msi, dice, jaccard, hausdorff in zip(
                    evaluation_metrics.idx,
                    evaluation_metrics.msi,
                    evaluation_metrics.dice,
                    evaluation_metrics.jaccard,
                    evaluation_metrics.hausdorff,
                ):
                    writer.writerow({
                        "test_segmentation": test_segmentation,
                        "slice_index": idx,
                        "include_zeros": False,
                        "msi": msi,
                        "dice": dice,
                        "jaccard": jaccard,
                        "hausdorff": hausdorff,
                    })

                # Metrics with zeros
                for idx, msi, dice, jaccard, hausdorff in zip(
                    evaluation_metrics_with_zeros.idx,
                    evaluation_metrics_with_zeros.msi,
                    evaluation_metrics_with_zeros.dice,
                    evaluation_metrics_with_zeros.jaccard,
                    evaluation_metrics_with_zeros.hausdorff,
                ):
                    writer.writerow({
                        "test_segmentation": test_segmentation,
                        "slice_index": idx,
                        "include_zeros": True,
                        "msi": msi,
                        "dice": dice,
                        "jaccard": jaccard,
                        "hausdorff": hausdorff,
                    })

